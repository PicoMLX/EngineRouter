import EngineRouter
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Synchronization
import Tokenizers

/// Runs MLX language models with mlx-swift-lm.
///
/// A model directory holds `config.json`, the tokenizer files and the weights
/// as `.safetensors` files, the layout `mlx-community` models use.
///
/// Each request runs on its own from an empty KV cache; requests to one model
/// are not batched together.
public struct MLXEngine: InferenceEngine {
    public let name = "mlx"
    public let isExclusive = false

    public init() {}

    public func canServe(_ model: EngineModel) async -> Bool {
        weightBytes(in: model.directory) != nil
    }

    /// The size of the weight files, which is what loading brings into memory.
    public func loadBytes(_ model: EngineModel) async throws -> Int {
        guard let bytes = weightBytes(in: model.directory) else {
            throw MLXEngineError.notAModel(model.directory.lastPathComponent)
        }
        return bytes
    }

    public func open(_ model: EngineModel) async throws -> any LoadedModel {
        let container = try await loadModelContainer(from: model.directory, using: #huggingFaceTokenizerLoader())
        return MLXModel(container: container, weightBytes: try await container.memory().total)
    }
}

/// The total size of the `.safetensors` files in `directory`, or `nil` when it
/// holds no `config.json` or no weights.
func weightBytes(in directory: URL) -> Int? {
    let files = (try? FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
    guard files.contains(where: { $0.lastPathComponent == "config.json" }) else { return nil }
    let weights = files.filter { $0.pathExtension == "safetensors" }
    guard !weights.isEmpty else { return nil }
    // A weight file can be a link, as in the Hugging Face cache; its size is its target's.
    return weights.reduce(0) {
        $0 + ((try? $1.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }
}

public enum MLXEngineError: Error, Equatable, CustomStringConvertible {
    /// The directory holds no `config.json` or no `.safetensors` weights.
    case notAModel(String)
    case closed

    public var description: String {
        switch self {
        case .notAModel(let directory): "\(directory) holds no MLX model."
        case .closed: "The MLX model is closed."
        }
    }
}

final class MLXModel: LoadedModel {
    private struct State {
        /// `nil` once the model is closed.
        var container: ModelContainer?
        var requests: [UUID: Task<Void, Never>] = [:]
    }

    private let weightBytes: Int
    private let state: Mutex<State>

    init(container: ModelContainer, weightBytes: Int) {
        self.weightBytes = weightBytes
        state = Mutex(State(container: container))
    }

    private func container() throws -> ModelContainer {
        guard let container = state.withLock({ $0.container }) else { throw MLXEngineError.closed }
        return container
    }

    // The app writes special tokens into the prompt itself, so the tokenizer
    // adds none and hides none.

    func tokenize(_ text: String) async throws -> [Int] {
        try await container().tokenizer.encode(text: text, addSpecialTokens: false)
    }

    func detokenize(_ tokens: [Int]) async throws -> String {
        try await container().tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
    }

    /// A request that sets no `temperature` or `topP` uses the model's own
    /// generation settings.
    func generate(_ request: EngineRequest) -> AsyncThrowingStream<EngineEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream<EngineEvent, any Error>.makeStream()
        let id = UUID()
        // The task is registered under the lock that `close` takes, so a
        // request either sees the model closed or is cancelled by `close`.
        let accepted = state.withLock { state in
            guard let container = state.container else { return false }
            state.requests[id] = Task { [self] in
                defer { self.state.withLock { $0.requests[id] = nil } }
                do {
                    try await Self.run(request, on: container, continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            return true
        }
        guard accepted else {
            continuation.finish(throwing: MLXEngineError.closed)
            return stream
        }
        continuation.onTermination = { [self] termination in
            if case .cancelled = termination { state.withLock { $0.requests[id] }?.cancel() }
        }
        return stream
    }

    private static func run(
        _ request: EngineRequest, on container: ModelContainer,
        _ continuation: AsyncThrowingStream<EngineEvent, any Error>.Continuation
    ) async throws {
        var settings = container.defaultGenerateParameters
        settings.maxTokens = request.maxTokens
        if let temperature = request.temperature { settings.temperature = temperature }
        if let topP = request.topP { settings.topP = topP }

        let parameters = settings
        let promptTokens = request.promptTokens
        let tokens = try await container.perform { context in
            try generateTokens(
                input: LMInput(tokens: MLXArray(promptTokens)), parameters: parameters, context: context)
        }
        // Leaving this loop, by cancellation too, ends the library's stream,
        // which stops its generation.
        for await event in tokens {
            switch event {
            case .token(let token):
                continuation.yield(.tokens([token]))
            case .info(let info):
                if info.stopReason == .cancelled { throw CancellationError() }
                continuation.yield(.finished(EngineCompletion(
                    reason: info.stopReason == .length ? .length : .stop,
                    promptTokenCount: info.promptTokenCount,
                    generatedTokenCount: info.generationTokenCount,
                    promptTime: info.promptTime, generationTime: info.generateTime)))
            }
        }
        try Task.checkCancellation()
    }

    /// MLX keeps one buffer cache for the whole process, so every open MLX
    /// model reports the same `cache`, and `peak` is the process's MLX peak.
    func memory() async -> EngineMemory {
        let active = state.withLock { $0.container != nil } ? weightBytes : 0
        return EngineMemory(active: active, cache: Memory.cacheMemory, peak: max(active, Memory.peakMemory))
    }

    /// Drops MLX's buffer cache, which is all or nothing.
    func trim(toFree bytes: Int) async -> Int {
        let cached = Memory.cacheMemory
        Memory.clearCache()
        return max(0, cached - Memory.cacheMemory)
    }

    func close() async {
        let running = state.withLock { state -> [Task<Void, Never>] in
            state.container = nil
            return Array(state.requests.values)
        }
        for request in running { request.cancel() }
        for request in running { await request.value }
        // The weights go back to MLX's buffer cache when the last reference
        // drops; clearing it returns them to the system.
        Memory.clearCache()
    }
}
