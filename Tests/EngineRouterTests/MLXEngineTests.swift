import EngineRouter
import Foundation
import Testing
@testable import EngineRouterMLX

struct MLXModelDirectoryTests {
    /// A directory holding files with these names and sizes, removed when the test ends.
    private func directory(_ files: [String: Int]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (name, size) in files {
            try Data(count: size).write(to: directory.appendingPathComponent(name))
        }
        return directory
    }

    @Test
    func aModelLoadsTheBytesOfItsWeightFiles() async throws {
        let directory = try directory([
            "config.json": 10, "tokenizer.json": 500,
            "model-00001-of-00002.safetensors": 3000, "model-00002-of-00002.safetensors": 1000,
        ])
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = EngineModel(id: "m", directory: directory)

        #expect(await MLXEngine().canServe(model))
        #expect(try await MLXEngine().loadBytes(model) == 4000)
    }

    /// The Hugging Face cache stores a snapshot's files as links into its blob store.
    @Test
    func aLinkedWeightFileCountsAtItsTargetsSize() async throws {
        let blobs = try directory(["weights-blob": 5000])
        let directory = try directory(["config.json": 10])
        defer {
            try? FileManager.default.removeItem(at: blobs)
            try? FileManager.default.removeItem(at: directory)
        }
        try FileManager.default.createSymbolicLink(
            at: directory.appendingPathComponent("model.safetensors"),
            withDestinationURL: blobs.appendingPathComponent("weights-blob"))

        #expect(try await MLXEngine().loadBytes(EngineModel(id: "m", directory: directory)) == 5000)
    }

    @Test(arguments: [
        ["model.safetensors": 100],  // no config
        ["config.json": 10, "model.gguf": 100],  // no safetensors weights
        [:],
    ])
    func otherDirectoriesAreNotMLXModels(files: [String: Int]) async throws {
        let directory = try directory(files)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = EngineModel(id: "m", directory: directory)

        #expect(await MLXEngine().canServe(model) == false)
        await #expect(throws: MLXEngineError.notAModel(directory.lastPathComponent)) {
            try await MLXEngine().loadBytes(model)
        }
    }
}

private enum LiveMLX {
    static var modelDirectory: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_MLX_MODEL_DIR"] }
}

/// Runs against a real MLX model. MLX needs its Metal library, which only an
/// Xcode build produces, so run it with `xcodebuild`, not `swift test`:
///
///     TEST_RUNNER_ENGINEROUTER_MLX_MODEL_DIR=/path/to/model \
///     xcodebuild test -scheme EngineRouter-Package -destination 'platform=macOS' \
///       -only-testing:EngineRouterTests/MLXLiveTests
@Suite(.enabled(if: LiveMLX.modelDirectory != nil), .serialized)
struct MLXLiveTests {
    private var model: EngineModel {
        get throws {
            EngineModel(id: "mlx-live", directory: URL(fileURLWithPath: try #require(LiveMLX.modelDirectory)))
        }
    }

    @Test(.timeLimit(.minutes(10)))
    func mlxMeetsTheEngineContract() async throws {
        let violations = await engineContractViolations(
            of: MLXEngine(), model: try model, prompt: "The capital of France is")

        #expect(violations == [])
    }

    @Test(.timeLimit(.minutes(10)))
    func theRouterGeneratesThroughMLX() async throws {
        let router = Router(engines: [MLXEngine()], memoryBudget: Int(ProcessInfo.processInfo.physicalMemory))
        let loaded = try await router.open(try model)
        let status = try #require(await router.status().first)
        #expect(status.engine == "mlx")
        #expect(status.memory?.active ?? 0 > 0)

        let prompt = try await loaded.tokenize("The capital of France is")
        var tokens: [Int] = []
        var completion: EngineCompletion?
        for try await event in loaded.generate(EngineRequest(promptTokens: prompt, maxTokens: 8, temperature: 0)) {
            switch event {
            case .tokens(let ids): tokens += ids
            case .finished(let finished): completion = finished
            }
        }
        print("MLX continued: \(try await loaded.detokenize(tokens))")
        #expect(completion?.generatedTokenCount == tokens.count)
        #expect(completion?.promptTokenCount == prompt.count)
        #expect(!tokens.isEmpty)

        await router.close("mlx-live")
        #expect(await router.status().isEmpty)
        await #expect(throws: MLXEngineError.closed) { try await loaded.tokenize("x") }
    }
}
