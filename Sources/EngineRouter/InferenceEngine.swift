import Foundation

/// A model on disk, as the router and its engines name it.
public struct EngineModel: Sendable, Hashable {
    /// The name requests use for this model. Unique within a router.
    public var id: String
    /// The directory holding the model's files, in whatever format its engine reads.
    public var directory: URL

    public init(id: String, directory: URL) {
        self.id = id
        self.directory = directory
    }
}

/// One inference runtime: MLX, Local Code, mlx-serve.
///
/// An engine keeps its own batching, scheduling and KV caches. The router only
/// decides which engine serves a model and whether the model fits in memory.
///
/// Loading weights and running forward passes block. An engine does that work
/// off its caller's executor, so an `await` on it never stalls the router.
public protocol InferenceEngine: Sendable {
    /// A short stable name, such as `"mlx"`, for logs and status.
    var name: String { get }

    /// Whether this engine sizes itself to the machine's free memory. A model of
    /// an exclusive engine opens only when no other model is open, and no other
    /// model opens beside it.
    var isExclusive: Bool { get }

    /// Whether this engine can run the model in `model.directory`. Reads
    /// metadata only; loads nothing.
    func canServe(_ model: EngineModel) async -> Bool

    /// The bytes `model` holds once open. The router charges this against its
    /// budget before anything is loaded.
    func loadBytes(_ model: EngineModel) async throws -> Int

    func open(_ model: EngineModel) async throws -> any LoadedModel
}

/// A model an engine has loaded and can generate with.
public protocol LoadedModel: Sendable {
    /// Token IDs for `text`. Special tokens written out in `text` become their
    /// IDs; nothing is added at either end.
    func tokenize(_ text: String) async throws -> [Int]

    func detokenize(_ tokens: [Int]) async throws -> String

    /// Runs one request. A request that runs to its end yields `.finished`
    /// last. Ending the stream early, by cancelling the consuming task or
    /// dropping the stream, cancels the request.
    func generate(_ request: EngineRequest) -> AsyncThrowingStream<EngineEvent, any Error>

    func memory() async -> EngineMemory

    /// Drops caches to free about `bytes`. Returns the bytes freed.
    func trim(toFree bytes: Int) async -> Int

    /// Cancels every running request and releases the model.
    func close() async
}

public struct EngineRequest: Sendable, Equatable {
    public var promptTokens: [Int]
    /// The most tokens to generate; `nil` leaves the limit to the engine.
    public var maxTokens: Int?
    /// `0` is greedy; `nil` is the engine's default.
    public var temperature: Float?
    /// `nil` is the engine's default.
    public var topP: Float?

    public init(promptTokens: [Int], maxTokens: Int? = nil, temperature: Float? = nil, topP: Float? = nil) {
        self.promptTokens = promptTokens
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
    }
}

public enum EngineEvent: Sendable, Equatable {
    /// Generated tokens, in order. The token that stopped generation is not among them.
    case tokens([Int])
    case finished(EngineCompletion)
}

public struct EngineCompletion: Sendable, Equatable {
    public enum Reason: Sendable, Equatable {
        /// The model produced a stop token.
        case stop
        /// The request's or the context's token limit was reached.
        case length
    }

    public var reason: Reason
    /// Prompt tokens the engine evaluated for this request.
    public var promptTokenCount: Int
    /// Prompt tokens the engine reused from a cache instead of evaluating.
    public var cachedPromptTokenCount: Int
    public var generatedTokenCount: Int
    /// Seconds from submission to the first generated token.
    public var promptTime: TimeInterval
    /// Seconds from the first generated token to the last.
    public var generationTime: TimeInterval
    /// The token that stopped the reply, when `reason` is `.stop` and the
    /// engine knows which it was. It is not among the generated tokens. Some
    /// chat formats end a reply with different tokens for different outcomes,
    /// a finished answer or a tool call for example, so the app may need it.
    public var stopToken: Int?

    public init(
        reason: Reason, promptTokenCount: Int, cachedPromptTokenCount: Int = 0, generatedTokenCount: Int,
        promptTime: TimeInterval, generationTime: TimeInterval, stopToken: Int? = nil
    ) {
        self.reason = reason
        self.promptTokenCount = promptTokenCount
        self.cachedPromptTokenCount = cachedPromptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.promptTime = promptTime
        self.generationTime = generationTime
        self.stopToken = stopToken
    }
}

/// Device memory one loaded model holds, in bytes.
public struct EngineMemory: Sendable, Equatable {
    /// Weights and live request state.
    public var active: Int
    /// Buffers kept for reuse, which `trim(toFree:)` can drop.
    public var cache: Int
    /// The highest `active` since the model opened.
    public var peak: Int

    public init(active: Int, cache: Int, peak: Int) {
        self.active = active
        self.cache = cache
        self.peak = peak
    }
}
