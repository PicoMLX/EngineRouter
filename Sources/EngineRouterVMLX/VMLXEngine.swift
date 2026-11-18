#if os(macOS)
import EngineRouter
import Foundation
import Synchronization

/// Runs MLX models with vmlx-swift, the engine the Osaurus app uses, in a
/// helper process.
///
/// vmlx-swift ships modules with the same names as the MLX libraries this
/// package's own MLX adapter links, so the two cannot be built into one
/// process. The helper is a separate executable that links vmlx-swift alone;
/// this adapter starts it and talks to it over its standard input and output.
///
/// Starting a process is not allowed inside the App Store sandbox. An app
/// there has to run the helper as an XPC service instead; see the README.
public struct VMLXEngine: InferenceEngine {
    public let name = "vmlx"
    public let isExclusive = false

    private let helper: Helper

    /// Starts the helper and checks that it speaks this adapter's protocol.
    ///
    /// - Parameter helperURL: The `enginerouter-vmlx-helper` executable, with
    ///   the resource bundle it was built with beside it.
    public init(helperURL: URL) async throws {
        let process = Process()
        process.executableURL = helperURL
        let toHelper = Pipe()
        let fromHelper = Pipe()
        process.standardInput = toHelper
        process.standardOutput = fromHelper
        do {
            try process.run()
        } catch {
            throw VMLXError.launch("The vmlx helper did not start: \(error.localizedDescription)")
        }
        try await self.init(
            connection: HelperConnection(
                toHelper: toHelper.fileHandleForWriting, fromHelper: fromHelper.fileHandleForReading),
            process: process)
    }

    /// An engine over a connection that is already open. Tests pass one whose
    /// other end stands in for the helper, and no process.
    init(connection: HelperConnection, process: Process? = nil) async throws {
        helper = Helper(connection: connection, process: process)
        let version = try await connection.call("hello").protocol
        guard version == HelperConnection.protocolVersion else {
            throw VMLXError.launch(
                "The vmlx helper speaks protocol \(version.map(String.init) ?? "unknown"); this adapter needs "
                    + "\(HelperConnection.protocolVersion). Rebuild it.")
        }
    }

    public func canServe(_ model: EngineModel) async -> Bool {
        (try? await helper.connection.call("canServe", ["directory": model.directory.path]).canServe) ?? false
    }

    public func loadBytes(_ model: EngineModel) async throws -> Int {
        guard let bytes = try await helper.connection.call("loadBytes", ["directory": model.directory.path]).loadBytes,
            bytes > 0
        else { throw VMLXError.helper("\(model.directory.lastPathComponent) holds no model vmlx runs.") }
        return bytes
    }

    public func open(_ model: EngineModel) async throws -> any LoadedModel {
        guard let handle = try await helper.connection.call("open", ["directory": model.directory.path]).model else {
            throw VMLXError.helper("The vmlx helper opened no model.")
        }
        return VMLXModel(helper: helper, handle: handle)
    }
}

/// The helper process, which runs for as long as an engine or one of its
/// models refers to it. Ending its input makes it close its models and exit.
final class Helper: Sendable {
    let connection: HelperConnection
    /// Kept so the process is collected when it exits.
    private let process: Mutex<Process?>

    init(connection: HelperConnection, process: Process?) {
        self.connection = connection
        self.process = Mutex(process)
    }

    deinit {
        connection.close()
    }
}

final class VMLXModel: LoadedModel {
    private let helper: Helper
    private let handle: Int
    /// Set by the first `close`; every later `close` waits for the same one.
    private let closing = Mutex<Task<Void, Never>?>(nil)

    init(helper: Helper, handle: Int) {
        self.helper = helper
        self.handle = handle
    }

    private func connection() throws -> HelperConnection {
        guard closing.withLock({ $0 == nil }) else { throw VMLXError.closed }
        return helper.connection
    }

    func tokenize(_ text: String) async throws -> [Int] {
        try await connection().call("tokenize", ["model": handle, "text": text]).tokens ?? []
    }

    func detokenize(_ tokens: [Int]) async throws -> String {
        try await connection().call("detokenize", ["model": handle, "tokens": tokens]).text ?? ""
    }

    /// A request that sets no `temperature` or `topP` uses the model's own
    /// generation settings. Requests to one model run together in the
    /// engine's batches.
    func generate(_ request: EngineRequest) -> AsyncThrowingStream<EngineEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream<EngineEvent, any Error>.makeStream()
        do {
            let connection = try connection()
            if let limit = request.maxTokens, limit <= 0 {
                // A limit of no tokens needs no engine.
                continuation.yield(.finished(EngineCompletion(
                    reason: .length, promptTokenCount: 0, generatedTokenCount: 0, promptTime: 0, generationTime: 0)))
                continuation.finish()
                return stream
            }
            let (id, replies) = try connection.stream("generate", [
                "model": handle, "prompt": request.promptTokens,
                "maxTokens": request.maxTokens.map { $0 as Any } ?? NSNull(),
                "temperature": request.temperature.map { Double($0) as Any } ?? NSNull(),
                "topP": request.topP.map { Double($0) as Any } ?? NSNull(),
            ])
            let forwarding = Task {
                do {
                    for try await reply in replies {
                        if let event = try Self.event(reply) { continuation.yield(event) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { termination in
                guard case .cancelled = termination else { return }
                // The helper answers with the message that ends the request.
                connection.notify("cancel", ["request": id])
                forwarding.cancel()
            }
        } catch {
            continuation.finish(throwing: error)
        }
        return stream
    }

    /// What one of the helper's messages for a generation means. The message
    /// that ends a cancelled or failed request throws.
    private static func event(_ reply: HelperReply) throws -> EngineEvent? {
        if let message = reply.error { throw VMLXError.helper(message) }
        switch reply.event {
        case "tokens":
            guard let tokens = reply.tokens, !tokens.isEmpty else { return nil }
            return .tokens(tokens)
        case "finished":
            return .finished(EngineCompletion(
                reason: reply.reason == "length" ? .length : .stop,
                promptTokenCount: reply.promptTokens ?? 0, cachedPromptTokenCount: reply.cachedPromptTokens ?? 0,
                generatedTokenCount: reply.generatedTokens ?? 0,
                promptTime: reply.promptSeconds ?? 0, generationTime: reply.generationSeconds ?? 0,
                stopToken: reply.reason == "length" ? nil : reply.stopToken))
        case "cancelled":
            throw CancellationError()
        default:
            throw VMLXError.helper("The vmlx helper sent an unknown event.")
        }
    }

    /// The helper process's MLX figures, which every model open in it shares.
    func memory() async -> EngineMemory {
        guard let result = try? await connection().call("memory", ["model": handle]) else {
            return EngineMemory(active: 0, cache: 0, peak: 0)
        }
        let active = result.active ?? 0
        return EngineMemory(active: active, cache: result.cache ?? 0, peak: max(active, result.peak ?? 0))
    }

    func trim(toFree bytes: Int) async -> Int {
        guard bytes > 0 else { return 0 }
        return (try? await connection().call("trim", ["model": handle, "bytes": bytes]).freed) ?? 0
    }

    func close() async {
        let closing = closing.withLock { closing in
            if let closing { return closing }
            let task = Task<Void, Never> { [helper, handle] in
                // The helper ends every request on the model, then lets it go.
                _ = try? await helper.connection.call("close", ["model": handle])
            }
            closing = task
            return task
        }
        // Every caller returns only once the model has let go.
        await closing.value
    }
}
#endif
