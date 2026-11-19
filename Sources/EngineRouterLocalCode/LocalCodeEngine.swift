#if os(macOS)
import EngineRouter
import Foundation
import Synchronization

/// Runs DeepSeek 4.1 checkpoints with Local Code's engine, loaded from a
/// dynamic library built from the Local Code fork. Its models are
/// ``ChatModel``s: Local Code formats the conversation and reads the reply.
///
/// Local Code is GPL-3.0. See the licence note in the README before shipping
/// an app that includes the library.
///
/// A model directory holds one `.engram` file, the `.ckpt` checkpoint with the
/// same base name, and optionally a DSpark drafter: a second `.ckpt` whose
/// name contains `dspark`.
public struct LocalCodeEngine: InferenceEngine {
    public let name = "local-code"
    /// Local Code sizes its resident weights to the memory that is free when
    /// the model loads, so it cannot share the machine with another model.
    public let isExclusive = true

    private let library: LocalCodeLibrary

    public var buildInfo: EngineBuildInfo? { library.buildInfo }

    /// - Parameter libraryURL: The `PicoLocalCodeNative.dylib` to load.
    public init(libraryURL: URL) throws {
        library = try LocalCodeLibrary(url: libraryURL)
    }

    public func canServe(_ model: EngineModel) async -> Bool {
        ModelFiles(directory: model.directory) != nil
    }

    public func loadBytes(_ model: EngineModel) async throws -> Int {
        let handle = try await openHandle(model)
        return try await library.onQueue { [library] in
            defer { library.close(handle.pointer) }
            return Int(library.residentBytes(handle.pointer))
        }
    }

    public func open(_ model: EngineModel) async throws -> any LoadedModel {
        let handle = try await openHandle(model)
        let bytes = try await library.onQueue { [library] in Int(library.residentBytes(handle.pointer)) }
        return LocalCodeModel(library: library, handle: handle, residentBytes: bytes)
    }

    private func openHandle(_ model: EngineModel) async throws -> ModelHandle {
        guard let files = ModelFiles(directory: model.directory) else {
            throw LocalCodeError.model("\(model.directory.lastPathComponent) holds no Local Code checkpoint.")
        }
        return try await library.onQueue { [library] in
            var message = [CChar](repeating: 0, count: 512)
            let pointer = files.checkpoint.path.withCString { checkpoint in
                files.engram.path.withCString { engram in
                    if let drafter = files.drafter {
                        drafter.path.withCString {
                            library.open(checkpoint, engram, $0, &message, Int32(message.count))
                        }
                    } else {
                        library.open(checkpoint, engram, nil, &message, Int32(message.count))
                    }
                }
            }
            guard let pointer else {
                throw LocalCodeError.model(string(fromUTF8: message))
            }
            return ModelHandle(pointer: pointer)
        }
    }
}

/// The text in a C buffer of UTF-8, up to its first NUL. `CChar` is signed, so
/// a byte above 127 is negative here and is reinterpreted, not converted.
func string(fromUTF8 buffer: [CChar]) -> String {
    String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

/// The library's opaque model handle. Used only on `LocalCodeLibrary.queue`.
struct ModelHandle: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer
}

struct ModelFiles: Equatable {
    let checkpoint: URL
    let engram: URL
    let drafter: URL?

    init?(directory: URL) {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)) ?? []
        let engrams = files.filter { $0.pathExtension == "engram" }
        guard engrams.count == 1 else { return nil }
        let checkpoint = engrams[0].deletingPathExtension().appendingPathExtension("ckpt")
        guard files.contains(where: { $0.lastPathComponent == checkpoint.lastPathComponent }) else { return nil }
        self.engram = engrams[0]
        self.checkpoint = checkpoint
        self.drafter = files
            .filter { $0.pathExtension == "ckpt" && $0.lastPathComponent.contains("dspark") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .first
    }
}

final class LocalCodeModel: ChatModel {
    private struct State {
        /// Set by the first `close`; every later `close` waits for the same one.
        var closing: Task<Void, Never>?
        /// Whether a generation has started, which is when the weights load.
        var hasGenerated = false
        /// How to cancel each running request.
        var requests: [ObjectIdentifier: @Sendable () -> Void] = [:]
    }

    /// The model name a chat request carries. The handle says which model it is.
    private static let requestModel = "local-code"

    private let library: LocalCodeLibrary
    private let handle: ModelHandle
    private let residentBytes: Int
    private let state = Mutex(State())

    init(library: LocalCodeLibrary, handle: ModelHandle, residentBytes: Int) {
        self.library = library
        self.handle = handle
        self.residentBytes = residentBytes
    }

    // The tokenizer is immutable static state in the library, so these two do
    // not wait behind a running generation.

    func tokenize(_ text: String) async throws -> [Int] {
        // Passed with its length, so a NUL character in the text is kept.
        let bytes = Array(text.utf8)
        let count = library.tokenize(bytes, Int32(bytes.count), nil, 0)
        guard count >= 0 else { throw LocalCodeError.invalidText }
        var tokens = [Int32](repeating: 0, count: Int(count))
        _ = library.tokenize(bytes, Int32(bytes.count), &tokens, count)
        return tokens.map(Int.init)
    }

    func detokenize(_ tokens: [Int]) async throws -> String {
        let ids = tokens.map { Int32(clamping: $0) }
        let length = library.detokenize(ids, Int32(ids.count), nil, 0)
        var text = [CChar](repeating: 0, count: Int(length) + 1)
        _ = library.detokenize(ids, Int32(ids.count), &text, Int32(text.count))
        // The text can hold NUL characters, so its length is the one the
        // library reported, not where the first NUL is.
        return String(decoding: text.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// `temperature` 0 picks the most likely token; any other value samples
    /// with the model's own settings. Local Code has no `topP`.
    func generate(_ request: EngineRequest) -> AsyncThrowingStream<EngineEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream<EngineEvent, any Error>.makeStream()
        if let limit = request.maxTokens, limit <= 0 {
            // A limit of no tokens ends the request before the library sees it:
            // the library reads 0 as "no limit", which is what `nil` asks for.
            if state.withLock({ $0.closing == nil }) {
                continuation.yield(.finished(EngineCompletion(
                    reason: .length, promptTokenCount: 0, generatedTokenCount: 0, promptTime: 0, generationTime: 0)))
                continuation.finish()
            } else {
                continuation.finish(throwing: LocalCodeError.closed)
            }
            return stream
        }
        let running = Request(continuation: continuation)
        let accepted = state.withLock { state in
            guard state.closing == nil else { return false }
            state.requests[ObjectIdentifier(running)] = running.cancel
            return true
        }
        guard accepted else {
            continuation.finish(throwing: LocalCodeError.closed)
            return stream
        }
        continuation.onTermination = { [running] termination in
            if case .cancelled = termination { running.cancel() }
        }

        let prompt = request.promptTokens.map { Int32(clamping: $0) }
        // 0 leaves the limit to the library.
        let maxTokens = Int32(clamping: request.maxTokens ?? 0)
        let deterministic: Int32 = request.temperature == 0 ? 1 : 0
        LocalCodeLibrary.queue.async { [self] in
            defer { state.withLock { $0.requests[ObjectIdentifier(running)] = nil } }
            guard !running.isCancelled else {
                continuation.finish(throwing: CancellationError())
                return
            }
            state.withLock { $0.hasGenerated = true }
            // Prompt tokens, generated tokens, prefill and decode microseconds, the stop token.
            var usage = [Int64](repeating: 0, count: 5)
            let status = library.generateTokens(
                handle.pointer, prompt, Int32(prompt.count), maxTokens, deterministic,
                { context, tokens, count in
                    Unmanaged<Request>.fromOpaque(context!).takeUnretainedValue().deliver(tokens, count)
                },
                Unmanaged.passUnretained(running).toOpaque(), &usage)
            switch status {
            case 0, 3:
                continuation.yield(.finished(EngineCompletion(
                    reason: status == 0 ? .stop : .length,
                    promptTokenCount: Int(usage[0]), generatedTokenCount: Int(usage[1]),
                    promptTime: Double(usage[2]) / 1_000_000, generationTime: Double(usage[3]) / 1_000_000,
                    stopToken: status == 0 && usage[4] >= 0 ? Int(usage[4]) : nil)))
                continuation.finish()
            case 1:
                continuation.finish(throwing: CancellationError())
            default:
                continuation.finish(throwing: LocalCodeError.generationFailed)
            }
        }
        return stream
    }

    func memory() async -> EngineMemory {
        let active = state.withLock { $0.hasGenerated && $0.closing == nil } ? residentBytes : 0
        return EngineMemory(active: active, cache: 0, peak: active)
    }

    /// Local Code lays the conversation out in the model's own chat format,
    /// tools included, and splits the reply into reasoning, text and tool calls
    /// itself. It keeps the KV caches of its last few conversations, so a turn
    /// that goes on from an earlier reply evaluates only what is new.
    /// `temperature` 0 picks the most likely token when the model is not
    /// thinking; Local Code has no other sampling setting.
    func chat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream<ChatEvent, any Error>.makeStream()
        guard let chat = library.chat else {
            continuation.finish(throwing: LocalCodeLibrary.noChat)
            return stream
        }
        let body: String
        do {
            body = String(decoding: try request.chatCompletionsBody(model: Self.requestModel), as: UTF8.self)
        } catch {
            continuation.finish(throwing: error)
            return stream
        }
        let turn = ChatTurn(continuation: continuation)
        let accepted = state.withLock { state in
            guard state.closing == nil else { return false }
            state.requests[ObjectIdentifier(turn)] = turn.cancel
            return true
        }
        guard accepted else {
            continuation.finish(throwing: LocalCodeError.closed)
            return stream
        }
        continuation.onTermination = { [turn] termination in
            if case .cancelled = termination { turn.cancel() }
        }
        LocalCodeLibrary.queue.async { [self] in
            defer { state.withLock { $0.requests[ObjectIdentifier(turn)] = nil } }
            guard !turn.isCancelled else {
                continuation.finish(throwing: CancellationError())
                return
            }
            state.withLock { $0.hasGenerated = true }
            let status = body.withCString { json in
                chat(
                    handle.pointer, json,
                    { context, chunk in
                        Unmanaged<ChatTurn>.fromOpaque(context!).takeUnretainedValue().receive(chunk)
                    },
                    Unmanaged.passUnretained(turn).toOpaque())
            }
            turn.finish(status: status)
        }
        return stream
    }

    func countPromptTokens(_ request: ChatRequest) async throws -> Int {
        guard let count = library.countChatTokens else { throw LocalCodeLibrary.noChat }
        let body = String(decoding: try request.chatCompletionsBody(model: Self.requestModel), as: UTF8.self)
        // Counting lays the prompt out without waiting behind a running turn.
        // Holding the state meanwhile keeps `close` from letting the handle go.
        return try state.withLock { state in
            guard state.closing == nil else { throw LocalCodeError.closed }
            var message = [CChar](repeating: 0, count: 512)
            let tokens = body.withCString { count(handle.pointer, $0, &message, Int32(message.count)) }
            guard tokens >= 0 else { throw LocalCodeError.model(string(fromUTF8: message)) }
            return Int(tokens)
        }
    }

    /// Lets go of the KV caches Local Code keeps for its recent conversations.
    /// That waits for a running turn to end.
    func trim(toFree bytes: Int) async -> Int {
        guard bytes > 0, let trim = library.trim else { return 0 }
        let freed = try? await library.onQueue { [self] () -> Int in
            // A close already queued runs after this, so the handle is still the model's.
            guard state.withLock({ $0.closing == nil }) else { return 0 }
            return Int(trim(handle.pointer))
        }
        return freed ?? 0
    }

    func close() async {
        let closing = state.withLock { state in
            if let closing = state.closing { return closing }
            for cancel in state.requests.values { cancel() }
            let closing = Task<Void, Never> { [library, handle] in
                // Queued behind the requests just cancelled, so none is still inside the library.
                _ = try? await library.onQueue { library.close(handle.pointer) }
            }
            state.closing = closing
            return closing
        }
        // Every caller returns only once the model has let go.
        await closing.value
    }
}

/// One running chat turn, shared with the library's callback.
private final class ChatTurn: @unchecked Sendable {
    private let cancelled = Mutex(false)
    private let continuation: AsyncThrowingStream<ChatEvent, any Error>.Continuation
    // The library's callback and `finish` run on the library's queue, one
    // after the other, so these two need no lock.
    private var reader = ChatCompletionsStream()
    private var failure: (any Error)?

    init(continuation: AsyncThrowingStream<ChatEvent, any Error>.Continuation) {
        self.continuation = continuation
    }

    var isCancelled: Bool { cancelled.withLock { $0 } }

    @Sendable func cancel() { cancelled.withLock { $0 = true } }

    /// Reads one chunk; a nil one only asks whether to go on. Returns 0 to
    /// stop the library's turn.
    func receive(_ chunk: UnsafePointer<CChar>?) -> Int32 {
        if let chunk, failure == nil {
            do {
                for event in try reader.events(fromChunk: Data(String(cString: chunk).utf8)) {
                    if case .terminated = continuation.yield(event) { cancel() }
                }
            } catch {
                failure = error
            }
        }
        return isCancelled || failure != nil ? 0 : 1
    }

    /// Ends the stream once the library has returned `status`.
    func finish(status: Int32) {
        if let failure {
            continuation.finish(throwing: failure)
        } else if status == 1 || isCancelled {
            continuation.finish(throwing: CancellationError())
        } else if status == 0 {
            continuation.finish(throwing: reader.isFinished ? nil : ChatError.incomplete)
        } else {
            continuation.finish(throwing: LocalCodeError.generationFailed)
        }
    }
}

/// One running request, shared with the library's callback.
private final class Request: Sendable {
    private let cancelled = Mutex(false)
    private let continuation: AsyncThrowingStream<EngineEvent, any Error>.Continuation

    init(continuation: AsyncThrowingStream<EngineEvent, any Error>.Continuation) {
        self.continuation = continuation
    }

    var isCancelled: Bool { cancelled.withLock { $0 } }

    @Sendable func cancel() { cancelled.withLock { $0 = true } }

    /// Returns 0 to stop the library's generation.
    func deliver(_ tokens: UnsafePointer<Int32>?, _ count: Int32) -> Int32 {
        if let tokens, count > 0 {
            let ids = UnsafeBufferPointer(start: tokens, count: Int(count)).map(Int.init)
            if case .terminated = continuation.yield(.tokens(ids)) { cancel() }
        }
        return isCancelled ? 0 : 1
    }
}
#endif
