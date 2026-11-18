#if os(macOS)
import EngineRouter
import Foundation
import Synchronization
import Testing
@testable import EngineRouterVMLX

/// What the adapter sent, as the helper would read it.
private struct SentRequest: Decodable, Sendable {
    var id: UInt64
    var op: String
    var directory: String?
    var model: Int?
    var text: String?
    var tokens: [Int]?
    var prompt: [Int]?
    var maxTokens: Int?
    var temperature: Double?
    var request: UInt64?
}

/// Stands in for the helper process: reads the adapter's frames from one pipe
/// and answers on another, as `respond` says.
private final class FakeHelper: Sendable {
    let connection: HelperConnection
    private let toAdapter: FileHandle
    private let sent = Mutex<[SentRequest]>([])

    /// - Parameter respond: Called on the fake's thread for each request, with
    ///   the fake itself to answer through. Requests it does not answer stay
    ///   in flight.
    init(respond: @escaping @Sendable (SentRequest, FakeHelper) -> Void) {
        let toHelper = Pipe()
        let fromHelper = Pipe()
        toAdapter = fromHelper.fileHandleForWriting
        connection = HelperConnection(
            toHelper: toHelper.fileHandleForWriting, fromHelper: fromHelper.fileHandleForReading)
        let reading = toHelper.fileHandleForReading
        Thread { [self] in
            while let frame = HelperConnection.readFrame(from: reading) {
                guard let request = try? JSONDecoder().decode(SentRequest.self, from: frame) else { continue }
                sent.withLock { $0.append(request) }
                respond(request, self)
            }
            exit()
        }.start()
    }

    var operations: [String] { sent.withLock { $0.map(\.op) } }
    var requests: [SentRequest] { sent.withLock { $0 } }

    /// Writes one message to the adapter.
    func send(_ json: String) {
        try? toAdapter.write(contentsOf: HelperConnection.framed(Data(json.utf8)))
    }

    /// The helper process ending.
    func exit() {
        try? toAdapter.close()
    }

    /// A helper that greets with protocol 1, opens model 7 and otherwise does what `respond` says.
    static func greeting(
        _ respond: @escaping @Sendable (SentRequest, FakeHelper) -> Void = { _, _ in }
    ) -> FakeHelper {
        FakeHelper { request, helper in
            switch request.op {
            case "hello": helper.send(#"{"id":\#(request.id),"result":{"protocol":1,"engine":"fake"}}"#)
            case "open": helper.send(#"{"id":\#(request.id),"result":{"model":7}}"#)
            default: respond(request, helper)
            }
        }
    }
}

struct VMLXEngineTests {
    private let model = EngineModel(id: "m", directory: URL(fileURLWithPath: "/models/m"))

    private func events(_ stream: AsyncThrowingStream<EngineEvent, any Error>) async throws -> [EngineEvent] {
        var events: [EngineEvent] = []
        for try await event in stream { events.append(event) }
        return events
    }

    @Test
    func aHelperThatSpeaksAnotherProtocolIsRefused() async {
        let helper = FakeHelper { request, helper in
            helper.send(#"{"id":\#(request.id),"result":{"protocol":2}}"#)
        }

        await #expect {
            _ = try await VMLXEngine(connection: helper.connection)
        } throws: { error in
            guard case VMLXError.launch = error else { return false }
            return true
        }
    }

    @Test
    func theHelperAnswersWhatTheEngineAsks() async throws {
        let helper = FakeHelper.greeting { request, helper in
            switch request.op {
            case "canServe": helper.send(#"{"id":\#(request.id),"result":{"canServe":true}}"#)
            case "loadBytes": helper.send(#"{"id":\#(request.id),"result":{"loadBytes":4000}}"#)
            // One token per UTF-8 byte of the text, which arrives whole across a NUL character.
            case "tokenize":
                let tokens = Array((request.text ?? "").utf8).map(String.init).joined(separator: ",")
                helper.send(#"{"id":\#(request.id),"result":{"tokens":[\#(tokens)]}}"#)
            case "detokenize": helper.send(#"{"id":\#(request.id),"result":{"text":"a\u0000b"}}"#)
            case "memory": helper.send(#"{"id":\#(request.id),"result":{"active":100,"cache":10,"peak":90}}"#)
            case "trim": helper.send(#"{"id":\#(request.id),"result":{"freed":10}}"#)
            default: break
            }
        }
        let engine = try await VMLXEngine(connection: helper.connection)

        #expect(await engine.canServe(model))
        #expect(try await engine.loadBytes(model) == 4000)
        let loaded = try await engine.open(model)
        #expect(try await loaded.tokenize("a\u{0}b") == [97, 0, 98])
        #expect(try await loaded.detokenize([1]) == "a\u{0}b")
        #expect(await loaded.memory() == EngineMemory(active: 100, cache: 10, peak: 100))
        #expect(await loaded.trim(toFree: 5) == 10)
        #expect(helper.requests.first { $0.op == "open" }?.directory == "/models/m")
        #expect(helper.requests.first { $0.op == "tokenize" }?.model == 7)
    }

    @Test
    func aRefusedRequestThrowsTheHelpersMessage() async throws {
        let helper = FakeHelper.greeting { request, helper in
            helper.send(#"{"id":\#(request.id),"error":"no such model"}"#)
        }
        let engine = try await VMLXEngine(connection: helper.connection)

        await #expect(throws: VMLXError.helper("no such model")) { try await engine.loadBytes(model) }
        #expect(await engine.canServe(model) == false)
    }

    @Test
    func aGenerationStreamsItsTokensThenItsCompletion() async throws {
        let helper = FakeHelper.greeting { request, helper in
            guard request.op == "generate" else { return }
            helper.send(#"{"id":\#(request.id),"event":"tokens","tokens":[5,6]}"#)
            helper.send(#"{"id":\#(request.id),"event":"tokens","tokens":[7]}"#)
            helper.send(
                #"{"id":\#(request.id),"event":"finished","reason":"stop","promptTokens":1,"cachedPromptTokens":2,"#
                    + #""generatedTokens":3,"promptSeconds":0.5,"generationSeconds":1.5,"stopToken":9}"#)
        }
        let loaded = try await VMLXEngine(connection: helper.connection).open(model)

        let events = try await events(
            loaded.generate(EngineRequest(promptTokens: [1, 2, 3], maxTokens: 8, temperature: 0)))

        #expect(events == [
            .tokens([5, 6]), .tokens([7]),
            .finished(EngineCompletion(
                reason: .stop, promptTokenCount: 1, cachedPromptTokenCount: 2, generatedTokenCount: 3,
                promptTime: 0.5, generationTime: 1.5, stopToken: 9)),
        ])
        let sent = try #require(helper.requests.first { $0.op == "generate" })
        #expect(sent.prompt == [1, 2, 3])
        #expect(sent.maxTokens == 8)
        #expect(sent.temperature == 0)
    }

    @Test
    func aLimitOfNoTokensFinishesWithoutReachingTheHelper() async throws {
        let helper = FakeHelper.greeting()
        let loaded = try await VMLXEngine(connection: helper.connection).open(model)

        let events = try await events(loaded.generate(EngineRequest(promptTokens: [1], maxTokens: 0)))

        #expect(events == [
            .finished(EngineCompletion(
                reason: .length, promptTokenCount: 0, generatedTokenCount: 0, promptTime: 0, generationTime: 0))
        ])
        #expect(!helper.operations.contains("generate"))
    }

    @Test(.timeLimit(.minutes(1)))
    func aConsumerThatGoesAwayCancelsItsRequestByID() async throws {
        let helper = FakeHelper.greeting { request, helper in
            switch request.op {
            case "generate": helper.send(#"{"id":\#(request.id),"event":"tokens","tokens":[5]}"#)
            case "cancel": helper.send(#"{"id":\#(request.request ?? 0),"event":"cancelled"}"#)
            default: break
            }
        }
        let loaded = try await VMLXEngine(connection: helper.connection).open(model)

        let consumer = Task {
            for try await _ in loaded.generate(EngineRequest(promptTokens: [1])) {
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        _ = try? await consumer.value
        while !helper.operations.contains("cancel") { await Task.yield() }

        let generate = try #require(helper.requests.first { $0.op == "generate" })
        #expect(helper.requests.first { $0.op == "cancel" }?.request == generate.id)
    }

    @Test(.timeLimit(.minutes(1)))
    func aHelperThatExitsFailsWhatIsInFlightAndWhatComesAfter() async throws {
        let helper = FakeHelper.greeting { request, helper in
            guard request.op == "generate" else { return }
            helper.send(#"{"id":\#(request.id),"event":"tokens","tokens":[5]}"#)
            helper.exit()
        }
        let loaded = try await VMLXEngine(connection: helper.connection).open(model)

        await #expect(throws: VMLXError.helperExited) {
            _ = try await events(loaded.generate(EngineRequest(promptTokens: [1])))
        }
        await #expect(throws: VMLXError.helperExited) { try await loaded.tokenize("x") }
    }

    @Test(.timeLimit(.minutes(1)))
    func everyCloseReturnsOnlyOnceTheHelperHasLetTheModelGo() async throws {
        let closeRequest = Mutex<UInt64?>(nil)
        let helper = FakeHelper.greeting { request, _ in
            if request.op == "close" { closeRequest.withLock { $0 = request.id } }
        }
        let loaded = try await VMLXEngine(connection: helper.connection).open(model)

        let first = Task { await loaded.close() }
        while closeRequest.withLock({ $0 }) == nil { await Task.yield() }
        let secondReturned = Mutex(false)
        let second = Task {
            await loaded.close()
            secondReturned.withLock { $0 = true }
        }
        try await Task.sleep(for: .milliseconds(100))
        #expect(!secondReturned.withLock { $0 }, "The helper is still closing the model.")

        helper.send(#"{"id":\#(closeRequest.withLock { $0 } ?? 0),"result":{}}"#)
        await first.value
        await second.value
        #expect(helper.operations.filter { $0 == "close" }.count == 1)
        await #expect(throws: VMLXError.closed) { try await loaded.tokenize("x") }
        await #expect(throws: VMLXError.closed) {
            _ = try await events(loaded.generate(EngineRequest(promptTokens: [1])))
        }
    }
}
private enum LiveVMLX {
    static var helper: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_VMLX_HELPER"] }
    static var modelDirectory: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_VMLX_MODEL_DIR"] }
}

/// Runs against the real helper process and an MLX model. Set both variables
/// to enable it:
///
///     ENGINEROUTER_VMLX_HELPER=/path/to/enginerouter-vmlx-helper \
///     ENGINEROUTER_VMLX_MODEL_DIR=/path/to/model swift test --filter VMLXLiveTests
@Suite(.enabled(if: LiveVMLX.helper != nil && LiveVMLX.modelDirectory != nil), .serialized)
struct VMLXLiveTests {
    private static let prompt = "The capital of France is"

    private var engine: VMLXEngine {
        get async throws { try await VMLXEngine(helperURL: URL(fileURLWithPath: try #require(LiveVMLX.helper))) }
    }

    private var model: EngineModel {
        get throws {
            EngineModel(id: "vmlx-live", directory: URL(fileURLWithPath: try #require(LiveVMLX.modelDirectory)))
        }
    }

    @Test(.timeLimit(.minutes(10)))
    func vmlxMeetsTheEngineContract() async throws {
        let violations = await engineContractViolations(of: try await engine, model: try model, prompt: Self.prompt)

        #expect(violations == [])
    }

    @Test(.timeLimit(.minutes(10)))
    func theRouterGeneratesThroughTheHelper() async throws {
        let router = Router(engines: [try await engine], memoryBudget: Int(ProcessInfo.processInfo.physicalMemory))
        let loaded = try await router.open(try model)
        #expect(await router.status().first?.engine == "vmlx")
        #expect(await loaded.memory().active > 0)

        let prompt = try await loaded.tokenize(Self.prompt)
        let request = EngineRequest(promptTokens: prompt, maxTokens: 8, temperature: 0)
        let alone = try await run(loaded, request)
        print("vmlx continued: \(try await loaded.detokenize(alone.tokens))")
        #expect(alone.tokens.count == 8)
        #expect(alone.completion?.reason == .length)

        // Two requests at once run in one batch and repeat the single run. The
        // prompt is now in the engine's prefix cache.
        async let first = run(loaded, request)
        async let second = run(loaded, request)
        let (one, two) = try await (first, second)
        #expect(one.tokens == alone.tokens)
        #expect(two.tokens == alone.tokens)
        let repeated = try #require(one.completion)
        #expect(repeated.cachedPromptTokenCount > 0)
        #expect(repeated.promptTokenCount + repeated.cachedPromptTokenCount == prompt.count)

        // A chat turn runs to the model's end token, which the completion names.
        // The ChatML markers are the ones Qwen models use.
        let endOfTurn = try await loaded.tokenize("<|im_end|>")
        #expect(endOfTurn.count == 1)
        let chat = try await loaded.tokenize(
            "<|im_start|>user\nReply with one word: what is the capital of France?<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        let turn = try await run(loaded, EngineRequest(promptTokens: chat, maxTokens: 64, temperature: 0))
        print("vmlx answered: \(try await loaded.detokenize(turn.tokens))")
        #expect(turn.completion?.reason == .stop)
        #expect(turn.completion?.stopToken == endOfTurn.first)
        #expect(!turn.tokens.contains(endOfTurn[0]))

        await router.close("vmlx-live")
        #expect(await router.status().isEmpty)
        await #expect(throws: VMLXError.closed) { try await loaded.tokenize("x") }
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
