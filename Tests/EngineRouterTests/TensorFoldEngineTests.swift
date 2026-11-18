#if os(macOS)
import EngineRouter
import Foundation
import Testing
@testable import EngineRouterTensorFold

struct TensorFoldEngineTests {
    @Test
    func aMissingLibraryIsReportedWhenTheEngineIsCreated() {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).dylib")

        #expect {
            try TensorFoldEngine(libraryURL: missing)
        } throws: { error in
            guard case TensorFoldError.library = error else { return false }
            return true
        }
    }
}

private enum LiveTensorFold {
    static var library: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_TENSORFOLD_LIBRARY"] }
    static var modelDirectory: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_TENSORFOLD_MODEL_DIR"] }
}

/// Runs against the real library and a model TensorFold serves. Set both
/// variables to enable it:
///
///     ENGINEROUTER_TENSORFOLD_LIBRARY=/path/to/libPicoTensorFold.dylib \
///     ENGINEROUTER_TENSORFOLD_MODEL_DIR=/path/to/model swift test --filter TensorFoldLiveTests
@Suite(.enabled(if: LiveTensorFold.library != nil && LiveTensorFold.modelDirectory != nil), .serialized)
struct TensorFoldLiveTests {
    private static let prompt = "The capital of France is"

    private var engine: TensorFoldEngine {
        get throws { try TensorFoldEngine(libraryURL: URL(fileURLWithPath: try #require(LiveTensorFold.library))) }
    }

    private var model: EngineModel {
        get throws {
            EngineModel(
                id: "tensorfold-live", directory: URL(fileURLWithPath: try #require(LiveTensorFold.modelDirectory)))
        }
    }

    @Test(.timeLimit(.minutes(10)))
    func tensorFoldMeetsTheEngineContract() async throws {
        let violations = await engineContractViolations(of: try engine, model: try model, prompt: Self.prompt)

        #expect(violations == [])
    }

    @Test(.timeLimit(.minutes(10)))
    func theRouterGeneratesThroughTensorFold() async throws {
        let router = Router(engines: [try engine], memoryBudget: Int(ProcessInfo.processInfo.physicalMemory))
        let loaded = try await router.open(try model)
        #expect(await router.status().first?.engine == "tensorfold")

        // A special token written in the text becomes one ID.
        let marker = try await loaded.tokenize("<|im_start|>")
        #expect(marker.count == 1)
        let prompt = try await loaded.tokenize(Self.prompt)

        // Two requests at once both finish, and greedy generation gives both the same tokens.
        let request = EngineRequest(promptTokens: prompt, maxTokens: 8, temperature: 0)
        async let first = run(loaded, request)
        async let second = run(loaded, request)
        let (one, two) = try await (first, second)
        print("TensorFold continued: \(try await loaded.detokenize(one.tokens))")
        #expect(one.tokens.count == 8)
        #expect(one.tokens == two.tokens)
        #expect(one.completion?.reason == .length)
        #expect(one.completion?.generatedTokenCount == 8)

        await router.close("tensorfold-live")
        #expect(await router.status().isEmpty)
        await #expect(throws: TensorFoldError.closed) { try await loaded.tokenize("x") }
    }

    private func run(
        _ loaded: any LoadedModel, _ request: EngineRequest
    ) async throws -> (tokens: [Int], completion: EngineCompletion?) {
        var tokens: [Int] = []
        var completion: EngineCompletion?
        for try await event in loaded.generate(request) {
            switch event {
            case .tokens(let ids): tokens += ids
            case .finished(let finished): completion = finished
            }
        }
        return (tokens, completion)
    }
}
#endif
