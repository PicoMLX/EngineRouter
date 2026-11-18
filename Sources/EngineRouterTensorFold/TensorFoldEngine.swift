#if os(macOS)
import EngineRouter
import Foundation
import Synchronization

/// Runs models with TensorFold's Metal engine, loaded from a dynamic library
/// built from the TensorFold fork.
///
/// A model directory is in the Hugging Face layout: `config.json`, the
/// tokenizer files and `.safetensors` weights. The library decides which
/// model families it serves.
public struct TensorFoldEngine: InferenceEngine {
    public let name = "tensorfold"
    /// TensorFold budgets its memory against the whole process and most of
    /// the machine, so it cannot share the machine with another engine's model.
    public let isExclusive = true

    private let library: TensorFoldLibrary

    /// - Parameter libraryURL: The `libPicoTensorFold.dylib` to load.
    public init(libraryURL: URL) throws {
        library = try TensorFoldLibrary(url: libraryURL)
    }

    public func canServe(_ model: EngineModel) async -> Bool {
        model.directory.path.withCString { library.canServe($0) } == 1
    }

    public func loadBytes(_ model: EngineModel) async throws -> Int {
        let bytes = model.directory.path.withCString { library.loadBytes($0) }
        guard bytes > 0 else {
            throw TensorFoldError.model("\(model.directory.lastPathComponent) holds no model TensorFold runs.")
        }
        return Int(bytes)
    }

    public func open(_ model: EngineModel) async throws -> any LoadedModel {
        let directory = model.directory.path
        let handle = try await library.onQueue { [library] in
            var message = [CChar](repeating: 0, count: 512)
            guard let pointer = directory.withCString({ library.open($0, &message, Int32(message.count)) }) else {
                throw TensorFoldError.model(string(fromUTF8: message))
            }
            return ModelHandle(pointer: pointer)
        }
        return TensorFoldModel(library: library, handle: handle)
    }
}

/// The text in a C buffer of UTF-8, up to its first NUL. `CChar` is signed, so
/// a byte above 127 is negative here and is reinterpreted, not converted.
func string(fromUTF8 buffer: [CChar]) -> String {
    String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

/// The library's opaque model handle. The library guards it with its own lock.
struct ModelHandle: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer
}

final class TensorFoldModel: LoadedModel {
    private struct State {
        var isClosed = false
        var nextID: UInt64 = 1
    }

    private let library: TensorFoldLibrary
    private let handle: ModelHandle
    /// Held across every library call that uses the handle, so none runs
    /// after `close` has freed it. Those calls return at once.
    private let state = Mutex(State())

    init(library: TensorFoldLibrary, handle: ModelHandle) {
        self.library = library
        self.handle = handle
    }

    /// Runs `body` with the handle, unless the model is closed.
    private func withHandle<T>(_ body: (UnsafeMutableRawPointer) throws -> T) throws -> T {
        try state.withLock { state in
            guard !state.isClosed else { throw TensorFoldError.closed }
            return try body(handle.pointer)
        }
    }

    func tokenize(_ text: String) async throws -> [Int] {
        try withHandle { pointer in
            try text.withCString { string in
                let count = library.tokenize(pointer, string, nil, 0)
                guard count >= 0 else { throw TensorFoldError.invalidText }
                var tokens = [Int32](repeating: 0, count: Int(count))
                _ = library.tokenize(pointer, string, &tokens, count)
                return tokens.map(Int.init)
            }
        }
    }

    func detokenize(_ tokens: [Int]) async throws -> String {
        try withHandle { pointer in
            let ids = tokens.map { Int32(clamping: $0) }
            let length = library.detokenize(pointer, ids, Int32(ids.count), nil, 0)
            guard length >= 0 else { throw TensorFoldError.invalidText }
            var text = [CChar](repeating: 0, count: Int(length) + 1)
            _ = library.detokenize(pointer, ids, Int32(ids.count), &text, Int32(text.count))
            return string(fromUTF8: text)
        }
    }

    /// A request that sets no `temperature`, or sets 0, picks the most likely
    /// token. Requests to one model run together in the engine's lanes.
    func generate(_ request: EngineRequest) -> AsyncThrowingStream<EngineEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream<EngineEvent, any Error>.makeStream()
        let prompt = request.promptTokens.map { Int32(clamping: $0) }
        let maxTokens = Int32(clamping: request.maxTokens ?? 0)
        // The library's final callback releases this reference.
        let context = Unmanaged.passRetained(Request(continuation: continuation))

        let submitted: Result<UInt64, TensorFoldError> = state.withLock { state in
            guard !state.isClosed else { return .failure(.closed) }
            let id = state.nextID
            state.nextID += 1
            // A negative value leaves the setting to the engine.
            let refused = library.submit(
                handle.pointer, id, prompt, Int32(prompt.count), maxTokens, request.temperature ?? -1,
                request.topP ?? -1,
                { context, tokens, count, status, usage in
                    let request = Unmanaged<Request>.fromOpaque(context!)
                    if status < 0 {
                        request.takeUnretainedValue().deliver(tokens, count)
                    } else {
                        request.takeRetainedValue().finish(status, usage)
                    }
                },
                context.toOpaque())
            return refused == 0 ? .success(id) : .failure(.generationFailed)
        }
        switch submitted {
        case .failure(let error):
            // A refused request gets no callback.
            context.release()
            continuation.finish(throwing: error)
        case .success(let id):
            continuation.onTermination = { [self] termination in
                // Reaches the engine unless the model has closed; closing cancels every request itself.
                if case .cancelled = termination { try? withHandle { library.cancel($0, id) } }
            }
        }
        return stream
    }

    func memory() async -> EngineMemory {
        var bytes = [Int64](repeating: 0, count: 3)
        guard (try? withHandle { library.memory($0, &bytes) }) == 0 else { return EngineMemory(active: 0, cache: 0, peak: 0) }
        return EngineMemory(active: Int(bytes[0]), cache: Int(bytes[1]), peak: Int(max(bytes[0], bytes[2])))
    }

    /// TensorFold has no call that drops its caches.
    func trim(toFree bytes: Int) async -> Int { 0 }

    func close() async {
        let wasOpen = state.withLock { state in
            defer { state.isClosed = true }
            return !state.isClosed
        }
        guard wasOpen else { return }
        // Cancels every request, delivers each one's final callback, and frees the model.
        try? await library.onQueue { [library, handle] in library.close(handle.pointer) }
    }
}

/// One running request, shared with the library's callback, which runs on
/// the engine's thread and must return at once.
private final class Request: Sendable {
    private let continuation: AsyncThrowingStream<EngineEvent, any Error>.Continuation

    init(continuation: AsyncThrowingStream<EngineEvent, any Error>.Continuation) {
        self.continuation = continuation
    }

    func deliver(_ tokens: UnsafePointer<Int32>?, _ count: Int32) {
        guard let tokens, count > 0 else { return }
        continuation.yield(.tokens(UnsafeBufferPointer(start: tokens, count: Int(count)).map(Int.init)))
    }

    /// `usage` holds five values: prompt tokens evaluated, prompt tokens
    /// taken from a cache, tokens generated, prefill and decode microseconds.
    func finish(_ status: Int32, _ usage: UnsafePointer<Int64>?) {
        switch status {
        case 0, 3:
            let usage = usage.map { Array(UnsafeBufferPointer(start: $0, count: 5)) } ?? [0, 0, 0, 0, 0]
            continuation.yield(.finished(EngineCompletion(
                reason: status == 0 ? .stop : .length,
                promptTokenCount: Int(usage[0]), cachedPromptTokenCount: Int(usage[1]),
                generatedTokenCount: Int(usage[2]),
                promptTime: Double(usage[3]) / 1_000_000, generationTime: Double(usage[4]) / 1_000_000)))
            continuation.finish()
        case 1:
            continuation.finish(throwing: CancellationError())
        default:
            continuation.finish(throwing: TensorFoldError.generationFailed)
        }
    }
}
#endif
