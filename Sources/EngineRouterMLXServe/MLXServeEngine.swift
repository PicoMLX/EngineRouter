#if os(macOS)
import EngineRouter
import Foundation
import Synchronization

/// Runs models with mlx-serve's MLX engine, loaded from a dynamic library
/// built from the mlx-serve fork. The library carries its own private copy of
/// MLX, which runs beside Swift MLX in the same process.
///
/// A model directory is in the Hugging Face layout: `config.json`, the
/// tokenizer files and `.safetensors` weights, including packs whose experts
/// are quantized as EXL3. The library decides which models it serves.
public struct MLXServeEngine: InferenceEngine {
    public let name = "mlx-serve"
    /// The library opens one model per process and does its own memory
    /// admission against the whole machine, so nothing else is open beside it.
    public let isExclusive = true

    private let library: MLXServeLibrary

    /// - Parameter libraryURL: The `libPicoMlxServe.dylib` to load. Its `mlx.metallib` sits beside it.
    public init(libraryURL: URL) throws {
        library = try MLXServeLibrary(url: libraryURL)
    }

    public func canServe(_ model: EngineModel) async -> Bool {
        model.directory.path.withCString { library.canServe($0) } == 1
    }

    public func loadBytes(_ model: EngineModel) async throws -> Int {
        let bytes = model.directory.path.withCString { library.loadBytes($0) }
        guard bytes > 0 else {
            throw MLXServeError.model("\(model.directory.lastPathComponent) holds no model mlx-serve runs.")
        }
        return Int(bytes)
    }

    public func open(_ model: EngineModel) async throws -> any LoadedModel {
        let directory = model.directory.path
        let handle = try await library.onQueue { [library] in
            var message = [CChar](repeating: 0, count: 512)
            guard let pointer = directory.withCString({ library.open($0, &message, Int32(message.count)) }) else {
                throw MLXServeError.model(string(fromUTF8: message))
            }
            return ModelHandle(pointer: pointer)
        }
        return MLXServeModel(library: library, handle: handle)
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

final class MLXServeModel: LoadedModel {
    private struct State {
        /// Set by the first `close`; every later `close` waits for the same one.
        var closing: Task<Void, Never>?
        var nextID: UInt64 = 1
    }

    private let library: MLXServeLibrary
    private let handle: ModelHandle
    /// Held across every library call that uses the handle, so none runs
    /// after `close` has freed it. Those calls return at once.
    private let state = Mutex(State())

    init(library: MLXServeLibrary, handle: ModelHandle) {
        self.library = library
        self.handle = handle
    }

    /// Runs `body` with the handle, unless the model is closed.
    private func withHandle<T>(_ body: (UnsafeMutableRawPointer) throws -> T) throws -> T {
        try state.withLock { state in
            guard state.closing == nil else { throw MLXServeError.closed }
            return try body(handle.pointer)
        }
    }

    func tokenize(_ text: String) async throws -> [Int] {
        // Passed with its length, so a NUL character in the text is kept.
        let bytes = Array(text.utf8)
        return try withHandle { pointer in
            let count = library.tokenize(pointer, bytes, Int32(bytes.count), nil, 0)
            guard count >= 0 else { throw MLXServeError.invalidText }
            var tokens = [Int32](repeating: 0, count: Int(count))
            _ = library.tokenize(pointer, bytes, Int32(bytes.count), &tokens, count)
            return tokens.map(Int.init)
        }
    }

    func detokenize(_ tokens: [Int]) async throws -> String {
        try withHandle { pointer in
            let ids = tokens.map { Int32(clamping: $0) }
            let length = library.detokenize(pointer, ids, Int32(ids.count), nil, 0)
            guard length >= 0 else { throw MLXServeError.invalidText }
            var text = [CChar](repeating: 0, count: Int(length) + 1)
            _ = library.detokenize(pointer, ids, Int32(ids.count), &text, Int32(text.count))
            // The text can hold NUL characters, so its length is the one the
            // library reported, not where the first NUL is.
            return String(decoding: text.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }
    }

    /// A request that sets no `temperature`, or sets 0, picks the most likely
    /// token. Requests to one model run together in the engine's scheduler.
    func generate(_ request: EngineRequest) -> AsyncThrowingStream<EngineEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream<EngineEvent, any Error>.makeStream()
        if let limit = request.maxTokens, limit <= 0 {
            // A limit of no tokens ends the request before the engine sees it:
            // the library reads 0 as "no limit", which is what `nil` asks for.
            if state.withLock({ $0.closing == nil }) {
                continuation.yield(.finished(EngineCompletion(
                    reason: .length, promptTokenCount: 0, generatedTokenCount: 0, promptTime: 0, generationTime: 0)))
                continuation.finish()
            } else {
                continuation.finish(throwing: MLXServeError.closed)
            }
            return stream
        }
        let prompt = request.promptTokens.map { Int32(clamping: $0) }
        // 0 leaves the limit to the engine.
        let maxTokens = Int32(clamping: request.maxTokens ?? 0)
        // The library's final callback releases this reference.
        let context = Unmanaged.passRetained(Request(continuation: continuation))

        let submitted: Result<UInt64, MLXServeError> = state.withLock { state in
            guard state.closing == nil else { return .failure(.closed) }
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

    /// Drops the engine's kept prompt prefixes and buffer pool, all or nothing.
    func trim(toFree bytes: Int) async -> Int {
        guard bytes > 0 else { return 0 }
        return (try? withHandle { Int(library.trim($0, Int64(bytes))) }) ?? 0
    }

    func close() async {
        let closing = state.withLock { state in
            if let closing = state.closing { return closing }
            let closing = Task<Void, Never> { [library, handle] in
                // Cancels every request, delivers each one's final callback, and frees the model.
                _ = try? await library.onQueue { library.close(handle.pointer) }
            }
            state.closing = closing
            return closing
        }
        // Every caller returns only once the model has let go.
        await closing.value
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

    /// `usage` holds six values: prompt tokens evaluated, prompt tokens
    /// taken from a cache, tokens generated, prefill and decode microseconds,
    /// and the end token that stopped the reply, or -1.
    func finish(_ status: Int32, _ usage: UnsafePointer<Int64>?) {
        switch status {
        case 0, 3:
            let usage = usage.map { Array(UnsafeBufferPointer(start: $0, count: 6)) } ?? [0, 0, 0, 0, 0, -1]
            continuation.yield(.finished(EngineCompletion(
                reason: status == 0 ? .stop : .length,
                promptTokenCount: Int(usage[0]), cachedPromptTokenCount: Int(usage[1]),
                generatedTokenCount: Int(usage[2]),
                promptTime: Double(usage[3]) / 1_000_000, generationTime: Double(usage[4]) / 1_000_000,
                stopToken: status == 0 && usage[5] >= 0 ? Int(usage[5]) : nil)))
            continuation.finish()
        case 1:
            continuation.finish(throwing: CancellationError())
        default:
            continuation.finish(throwing: MLXServeError.generationFailed)
        }
    }
}
#endif
