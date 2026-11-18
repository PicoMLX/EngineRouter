#if os(macOS)
import Dispatch
import EngineRouter
import Foundation
import Synchronization
import Testing
@testable import EngineRouterMLXServe

struct MLXServeEngineTests {
    @Test
    func aMissingLibraryIsReportedWhenTheEngineIsCreated() {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).dylib")

        #expect {
            try MLXServeEngine(libraryURL: missing)
        } throws: { error in
            guard case MLXServeError.library = error else { return false }
            return true
        }
    }
}

/// A model over entry points that stand in for the library. C function
/// pointers capture nothing, so what they record lives in `FakeLibrary`.
@Suite(.serialized)
struct MLXServeModelTests {
    private enum FakeLibrary {
        static let submits = Mutex(0)
        /// What the fake engine reports: 4 tokens generated, stopped by token 9.
        static let usage: [Int64] = [1, 0, 4, 10, 20, 9]
        static let closeMayFinish = DispatchSemaphore(value: 0)
        /// What the fake detokenizer writes: text with a NUL character inside it.
        static let text: [UInt8] = Array("a\u{0}b".utf8)
    }

    private func model(close: @escaping MLXServeLibrary.Close = { _ in }) -> MLXServeModel {
        let library = MLXServeLibrary(
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
            // Stops at once, or at a limit of 2 runs to it; either way it names token 9.
            submit: { _, _, _, _, maxTokens, _, _, callback, context in
                FakeLibrary.submits.withLock { $0 += 1 }
                FakeLibrary.usage.withUnsafeBufferPointer { callback?(context, nil, 0, maxTokens == 2 ? 3 : 0, $0.baseAddress) }
                return 0
            },
            cancel: { _, _ in }, memory: { _, _ in 1 }, trim: { _, _ in 0 })
        return MLXServeModel(library: library, handle: ModelHandle(pointer: UnsafeMutableRawPointer(bitPattern: 1)!))
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

    @Test
    func aReplyThatStoppedNamesItsStopTokenAndOneThatHitTheLimitNamesNone() async throws {
        let model = model()
        var completions: [EngineCompletion] = []
        for limit in [nil, 2] {
            for try await event in model.generate(EngineRequest(promptTokens: [1], maxTokens: limit)) {
                if case .finished(let completion) = event { completions.append(completion) }
            }
        }

        #expect(completions.map(\.reason) == [.stop, .length])
        #expect(completions.map(\.stopToken) == [9, nil])
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

private enum LiveMLXServe {
    static var library: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_MLXSERVE_LIBRARY"] }
    static var modelDirectory: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_MLXSERVE_MODEL_DIR"] }
}

/// Runs against the real library and a model mlx-serve runs. Set both
/// variables to enable it:
///
///     ENGINEROUTER_MLXSERVE_LIBRARY=/path/to/libPicoMlxServe.dylib \
///     ENGINEROUTER_MLXSERVE_MODEL_DIR=/path/to/model swift test --filter MLXServeLiveTests
@Suite(.enabled(if: LiveMLXServe.library != nil && LiveMLXServe.modelDirectory != nil), .serialized)
struct MLXServeLiveTests {
    private static let prompt = "The capital of France is"

    private var engine: MLXServeEngine {
        get throws { try MLXServeEngine(libraryURL: URL(fileURLWithPath: try #require(LiveMLXServe.library))) }
    }

    private var model: EngineModel {
        get throws {
            EngineModel(
                id: "mlx-serve-live", directory: URL(fileURLWithPath: try #require(LiveMLXServe.modelDirectory)))
        }
    }

    @Test(.timeLimit(.minutes(10)))
    func mlxServeMeetsTheEngineContract() async throws {
        let violations = await engineContractViolations(of: try engine, model: try model, prompt: Self.prompt)

        #expect(violations == [])
    }

    @Test(.timeLimit(.minutes(10)))
    func theRouterGeneratesThroughMLXServe() async throws {
        let router = Router(engines: [try engine], memoryBudget: Int(ProcessInfo.processInfo.physicalMemory))
        let loaded = try await router.open(try model)
        #expect(await router.status().first?.engine == "mlx-serve")

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
        print("mlx-serve continued: \(try await loaded.detokenize(one.tokens))")
        #expect(one.tokens.count == 8)
        #expect(one.tokens == two.tokens)
        #expect(one.completion?.reason == .length)
        #expect(one.completion?.generatedTokenCount == 8)

        // A chat turn runs to the model's end token, which the completion names.
        // The ChatML markers are the ones Qwen models use.
        let endOfTurn = try await loaded.tokenize("<|im_end|>")
        #expect(endOfTurn.count == 1)
        let chat = try await loaded.tokenize(
            "<|im_start|>user\nReply with one word: what is the capital of France?<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        let turn = try await run(loaded, EngineRequest(promptTokens: chat, maxTokens: 64, temperature: 0))
        print("mlx-serve answered: \(try await loaded.detokenize(turn.tokens))")
        #expect(turn.completion?.reason == .stop)
        #expect(turn.completion?.stopToken == endOfTurn.first)
        #expect(!turn.tokens.contains(endOfTurn[0]))

        // Dropping kept prompt prefixes frees memory.
        #expect(await loaded.trim(toFree: 1) >= 0)

        await router.close("mlx-serve-live")
        #expect(await router.status().isEmpty)
        await #expect(throws: MLXServeError.closed) { try await loaded.tokenize("x") }
    }

    /// Holds for any model: one whose reply stops within the limit shows the stop token.
    @Test(.timeLimit(.minutes(10)))
    func onlyAReplyThatStopsNamesAStopToken() async throws {
        let loaded = try await engine.open(try model)
        let prompt = try await loaded.tokenize(Self.prompt)
        let (tokens, completion) = try await run(
            loaded, EngineRequest(promptTokens: prompt, maxTokens: 32, temperature: 0))
        await loaded.close()

        let finished = try #require(completion)
        print("mlx-serve finished with \(finished.reason) after \(tokens.count) tokens, stop token \(String(describing: finished.stopToken))")
        if finished.reason == .stop {
            let stopToken = try #require(finished.stopToken)
            #expect(!tokens.contains(stopToken))
        } else {
            #expect(finished.stopToken == nil)
        }
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
