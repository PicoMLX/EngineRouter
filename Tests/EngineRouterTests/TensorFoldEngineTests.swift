#if os(macOS)
import Dispatch
import EngineRouter
import Foundation
import Synchronization
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

/// A model over entry points that stand in for the library. C function
/// pointers capture nothing, so what they record lives in `FakeLibrary`.
@Suite(.serialized)
struct TensorFoldModelTests {
    private enum FakeLibrary {
        static let submits = Mutex(0)
        static let closeMayFinish = DispatchSemaphore(value: 0)
        /// What the fake detokenizer writes: text with a NUL character inside it.
        static let text: [UInt8] = Array("a\u{0}b".utf8)
    }

    private func model(close: @escaping TensorFoldLibrary.Close = { _ in }) -> TensorFoldModel {
        let library = TensorFoldLibrary(
            canServe: { _ in 1 }, loadBytes: { _ in 1 }, open: { _, _, _ in nil }, close: close,
            // One token per byte of the text it is handed.
            tokenize: { _, text, length, out, capacity in
                if let text, let out {
                    for index in 0..<Int(min(length, capacity)) { out[index] = Int32(text[index]) }
                }
                return length
            },
            detokenize: { _, _, _, out, capacity in
                let text = FakeLibrary.text
                if let out, capacity > 0 {
                    let written = min(text.count, Int(capacity) - 1)
                    for index in 0..<written { out[index] = CChar(bitPattern: text[index]) }
                    out[written] = 0
                }
                return Int32(text.count)
            },
            submit: { _, _, _, _, _, _, _, _, _ in
                FakeLibrary.submits.withLock { $0 += 1 }
                return 1
            },
            cancel: { _, _ in }, memory: { _, _ in 1 })
        return TensorFoldModel(library: library, handle: ModelHandle(pointer: UnsafeMutableRawPointer(bitPattern: 1)!))
    }

    @Test
    func theWholeTextReachesTheTokenizerAcrossANULCharacter() async throws {
        #expect(try await model().tokenize("a\u{0}b") == [97, 0, 98])
    }

    @Test
    func detokenizedTextKeepsANULCharacterAndWhatFollowsIt() async throws {
        #expect(try await model().detokenize([1, 2, 3]) == "a\u{0}b")
    }

    @Test
    func aLimitOfNoTokensFinishesWithoutReachingTheEngine() async throws {
        FakeLibrary.submits.withLock { $0 = 0 }
        var events: [EngineEvent] = []
        for try await event in model().generate(EngineRequest(promptTokens: [1, 2], maxTokens: 0)) {
            events.append(event)
        }

        #expect(events == [
            .finished(EngineCompletion(
                reason: .length, promptTokenCount: 0, generatedTokenCount: 0, promptTime: 0, generationTime: 0))
        ])
        #expect(FakeLibrary.submits.withLock { $0 } == 0)
    }

    @Test(.timeLimit(.minutes(1)))
    func everyCloseReturnsOnlyOnceTheModelHasLetGo() async throws {
        let model = model(close: { _ in FakeLibrary.closeMayFinish.wait() })
        let first = Task { await model.close() }
        // The second close starts once the first has marked the model closed.
        while (try? await model.tokenize("x")) != nil { await Task.yield() }
        let secondReturned = Mutex(false)
        let second = Task {
            await model.close()
            secondReturned.withLock { $0 = true }
        }

        try await Task.sleep(for: .milliseconds(100))
        #expect(!secondReturned.withLock { $0 }, "The library is still closing the model.")
        FakeLibrary.closeMayFinish.signal()
        await first.value
        await second.value
        #expect(secondReturned.withLock { $0 })
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
        // A NUL character is text like any other.
        let withNUL = "a\u{0}b"
        #expect(try await loaded.detokenize(try await loaded.tokenize(withNUL)) == withNUL)

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
