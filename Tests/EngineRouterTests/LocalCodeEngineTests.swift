#if os(macOS)
import Dispatch
import EngineRouter
import Foundation
import Synchronization
import Testing
@testable import EngineRouterLocalCode

struct LocalCodeModelFilesTests {
    /// A directory holding empty files with these names, removed when the test ends.
    private func directory(_ names: [String]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for name in names { try Data().write(to: directory.appendingPathComponent(name)) }
        return directory
    }

    @Test
    func aCheckpointIsNamedAfterItsEngramFile() throws {
        let directory = try directory(["model_i2x.ckpt", "model_i2x.engram", "notes.txt"])
        defer { try? FileManager.default.removeItem(at: directory) }

        let files = try #require(ModelFiles(directory: directory))

        #expect(files.checkpoint.lastPathComponent == "model_i2x.ckpt")
        #expect(files.engram?.lastPathComponent == "model_i2x.engram")
        #expect(files.drafter == nil)
        #expect(files.servesTokens)
    }

    @Test
    func aSecondCheckpointNamedDSparkIsTheDrafter() throws {
        let directory = try directory(["model_i2x.ckpt", "model_i2x.engram", "model_dspark_i4x.ckpt"])
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(ModelFiles(directory: directory)?.drafter?.lastPathComponent == "model_dspark_i4x.ckpt")
    }

    @Test
    func aGLMCheckpointNeedsNoEngramFileAndIsServedThroughChatOnly() throws {
        let directory = try directory(["glm_5.3_flash_i4x.ckpt", "notes.txt"])
        defer { try? FileManager.default.removeItem(at: directory) }

        let files = try #require(ModelFiles(directory: directory))

        #expect(files.checkpoint.lastPathComponent == "glm_5.3_flash_i4x.ckpt")
        #expect(files.engram == nil)
        #expect(files.drafter == nil)
        #expect(!files.servesTokens)
    }

    @Test
    func aDFlashCheckpointBesideGLMIsItsDrafter() throws {
        let directory = try directory(["glm_5.3_flash_i4x.ckpt", "glm_5.3_flash_dflash2_i5x.ckpt"])
        defer { try? FileManager.default.removeItem(at: directory) }

        let files = try #require(ModelFiles(directory: directory))

        #expect(files.checkpoint.lastPathComponent == "glm_5.3_flash_i4x.ckpt")
        #expect(files.drafter?.lastPathComponent == "glm_5.3_flash_dflash2_i5x.ckpt")
    }

    @Test(arguments: [
        ["model.ckpt"],  // no engram file
        ["model.engram"],  // no checkpoint
        ["a.engram", "b.ckpt"],  // names do not match
        ["a.engram", "a.ckpt", "b.engram"],  // two engram files
        ["glm_5.3_flash_i4x.ckpt.78e1a4.xet-partial"],  // a download still under way
        ["glm_5.3_flash_i4x.ckpt", "glm_5.3_flash_i2x.ckpt"],  // two GLM checkpoints
        ["glm_5.3_flash_dflash2_i5x.ckpt"],  // a drafter alone
    ])
    func otherDirectoriesAreNotLocalCodeModels(names: [String]) throws {
        let directory = try directory(names)
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(ModelFiles(directory: directory) == nil)
    }

    @Test
    func textFromTheLibraryKeepsItsNonASCIICharacters() {
        let text = "café 日本語 🙂 — done"
        // What the library writes: UTF-8 as signed C characters, NUL-terminated.
        let buffer = text.utf8CString.map { $0 } + [0x41]

        #expect(string(fromUTF8: buffer) == text)
    }

    @Test
    func aMissingLibraryIsReportedWhenTheEngineIsCreated() {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).dylib")

        #expect(throws: LocalCodeError.self) { try LocalCodeEngine(libraryURL: missing) }
    }
}

/// A model over entry points that stand in for the library. C function
/// pointers capture nothing, so what they record lives in `FakeLibrary`.
@Suite(.serialized)
struct LocalCodeModelTests {
    private enum FakeLibrary {
        static let generations = Mutex(0)
        static let closeStarted = DispatchSemaphore(value: 0)
        static let closeMayFinish = DispatchSemaphore(value: 0)
        /// What the fake detokenizer writes: text with a NUL character inside it.
        static let text: [UInt8] = Array("a\u{0}b".utf8)
        /// What the fake chat entry point streams: thinking, a tool call, the end and the usage.
        static let chatChunks = [
            #"{"choices":[{"index":0,"delta":{"reasoning_content":"Look it up."},"finish_reason":null}]}"#,
            #"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","#
                + #""function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]}"#,
            #"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            #"{"choices":[],"usage":{"prompt_tokens":20,"prompt_tokens_details":{"cached_tokens":12},"#
                + #""completion_tokens":9}}"#,
        ]
        static let chatRequests = Mutex<[String]>([])
        static let chatStoppedByCallback = Mutex(false)
    }

    private func model(
        close: @escaping LocalCodeLibrary.Close = { _ in }, chat: LocalCodeLibrary.Chat? = nil,
        countChatTokens: LocalCodeLibrary.CountChatTokens? = nil, servesTokens: Bool = true
    ) -> LocalCodeModel {
        let library = LocalCodeLibrary(
            open: { _, _, _, _, _ in nil }, close: close, residentBytes: { _ in 1 },
            // One token per byte of the text it is handed.
            tokenize: { text, length, out, capacity in
                if let text, let out {
                    for index in 0..<Int(min(length, capacity)) { out[index] = Int32(text[index]) }
                }
                return length
            },
            detokenize: { _, _, out, capacity in
                let text = FakeLibrary.text
                if let out, capacity > 0 {
                    let written = min(text.count, Int(capacity) - 1)
                    for index in 0..<written { out[index] = CChar(bitPattern: text[index]) }
                    out[written] = 0
                }
                return Int32(text.count)
            },
            // Stops at token 7, or runs to a limit of 2; either way it writes 7 as the stop token.
            generateTokens: { _, _, _, maxTokens, _, _, _, usage in
                FakeLibrary.generations.withLock { $0 += 1 }
                usage?[4] = 7
                return maxTokens == 2 ? 3 : 0
            },
            chat: chat, countChatTokens: countChatTokens)
        return LocalCodeModel(
            library: library, handle: ModelHandle(pointer: UnsafeMutableRawPointer(bitPattern: 1)!), residentBytes: 1,
            servesTokens: servesTokens)
    }

    /// GLM 5.3 Flash's tokens are not the DeepSeek 4.1 tokens the library's
    /// token-level calls speak, so those calls refuse it rather than answer wrongly.
    @Test
    func aChatOnlyModelRefusesTheTokenLevelCalls() async throws {
        let model = model(servesTokens: false)

        await #expect(throws: LocalCodeError.chatOnly) { try await model.tokenize("a") }
        await #expect(throws: LocalCodeError.chatOnly) { try await model.detokenize([1]) }
        await #expect(throws: LocalCodeError.chatOnly) {
            for try await _ in model.generate(EngineRequest(promptTokens: [1])) {}
        }
    }

    @Test
    func aChatTurnArrivesAsTheEventsTheLibraryStreams() async throws {
        let model = model(chat: { _, json, callback, context in
            FakeLibrary.chatRequests.withLock { $0.append(json.map { String(cString: $0) } ?? "") }
            for chunk in FakeLibrary.chatChunks {
                guard chunk.withCString({ callback?(context, $0) }) != 0 else { return 1 }
            }
            return 0
        })
        let request = ChatRequest(
            messages: [ChatMessage(role: .user, content: "Weather in Paris?")],
            tools: [ChatTool(name: "weather", parameters: #"{"type":"object"}"#)])

        var events: [ChatEvent] = []
        for try await event in model.chat(request) { events.append(event) }

        #expect(Array(events.dropLast()) == [
            .reasoning("Look it up."),
            .toolCall(ChatToolCall(id: "call_x", name: "weather", arguments: #"{"city":"Paris"}"#)),
        ])
        guard case .finished(let completion) = events.last else {
            Issue.record("The turn did not finish.")
            return
        }
        #expect(completion.reason == .toolCalls)
        #expect(completion.promptTokenCount == 20)
        #expect(completion.cachedPromptTokenCount == 12)
        // The library gets the conversation and the tools as a Chat Completions body.
        let sent = try #require(FakeLibrary.chatRequests.withLock { $0.last }.flatMap { $0.data(using: .utf8) })
        let body = try #require(try JSONSerialization.jsonObject(with: sent) as? [String: Any])
        #expect((body["messages"] as? [[String: Any]])?.first?["content"] as? String == "Weather in Paris?")
        #expect((body["tools"] as? [[String: Any]])?.count == 1)
    }

    @Test(.timeLimit(.minutes(1)))
    func aConsumerThatStopsReadingStopsTheLibrarysTurn() async throws {
        let model = model(chat: { _, _, callback, context in
            _ = FakeLibrary.chatChunks[0].withCString { callback?(context, $0) }
            // Asks whether to go on until the answer is no.
            while callback?(context, nil) != 0 { usleep(1000) }
            FakeLibrary.chatStoppedByCallback.withLock { $0 = true }
            return 1
        })

        for try await _ in model.chat(ChatRequest(messages: [ChatMessage(role: .user, content: "Hi")])) { break }

        while !FakeLibrary.chatStoppedByCallback.withLock({ $0 }) { try await Task.sleep(for: .milliseconds(10)) }
    }

    @Test
    func aTurnTheLibraryEndsWithoutItsLastChunkIsIncomplete() async {
        let model = model(chat: { _, _, callback, context in
            _ = FakeLibrary.chatChunks[0].withCString { callback?(context, $0) }
            return 0
        })

        await #expect(throws: ChatError.incomplete) {
            for try await _ in model.chat(ChatRequest(messages: [ChatMessage(role: .user, content: "Hi")])) {}
        }
    }

    @Test
    func anErrorTheLibraryWritesIsTheTurnsError() async {
        let model = model(chat: { _, _, callback, context in
            _ = #"{"error":{"code":"invalid_request","message":"Too long."}}"#.withCString { callback?(context, $0) }
            return 2
        })

        await #expect(throws: ChatError.engine("Too long.")) {
            for try await _ in model.chat(ChatRequest(messages: [ChatMessage(role: .user, content: "Hi")])) {}
        }
    }

    @Test
    func countingPromptTokensAsksTheLibrary() async throws {
        let model = model(countChatTokens: { _, json, _, _ in json == nil ? -1 : 42 })

        #expect(try await model.countPromptTokens(ChatRequest(messages: [ChatMessage(role: .user, content: "Hi")])) == 42)
    }

    @Test
    func aLibraryBuiltBeforeTheChatEntryPointsSaysToRebuildIt() async {
        let model = model()
        let request = ChatRequest(messages: [ChatMessage(role: .user, content: "Hi")])

        await #expect(throws: LocalCodeLibrary.noChat) {
            for try await _ in model.chat(request) {}
        }
        await #expect(throws: LocalCodeLibrary.noChat) { try await model.countPromptTokens(request) }
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
    func aLimitOfNoTokensFinishesWithoutReachingTheLibrary() async throws {
        FakeLibrary.generations.withLock { $0 = 0 }
        var events: [EngineEvent] = []
        for try await event in model().generate(EngineRequest(promptTokens: [1, 2], maxTokens: 0)) {
            events.append(event)
        }

        #expect(events == [
            .finished(EngineCompletion(
                reason: .length, promptTokenCount: 0, generatedTokenCount: 0, promptTime: 0, generationTime: 0))
        ])
        #expect(FakeLibrary.generations.withLock { $0 } == 0)
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
        #expect(completions.map(\.stopToken) == [7, nil])
    }

    @Test(.timeLimit(.minutes(1)))
    func everyCloseReturnsOnlyOnceTheModelHasLetGo() async throws {
        let model = model(close: { _ in
            FakeLibrary.closeStarted.signal()
            FakeLibrary.closeMayFinish.wait()
        })
        let first = Task { await model.close() }
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                FakeLibrary.closeStarted.wait()
                continuation.resume()
            }
        }
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

private enum LiveLocalCode {
    static var library: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_LOCAL_CODE_LIBRARY"] }
    static var modelDirectory: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_LOCAL_CODE_MODEL_DIR"] }
    /// A directory holding a GLM 5.3 Flash checkpoint, and no Engram file.
    static var glmDirectory: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_LOCAL_CODE_GLM_DIR"] }
}

/// One chat turn's text, tool calls and completion.
private func chatTurn(
    _ model: any ChatModel, _ request: ChatRequest
) async throws -> (text: String, calls: [ChatToolCall], completion: ChatCompletion) {
    var text = ""
    var calls: [ChatToolCall] = []
    var completion: ChatCompletion?
    for try await event in model.chat(request) {
        switch event {
        case .text(let piece): text += piece
        case .reasoning: break
        case .toolCall(let call): calls.append(call)
        case .finished(let finished): completion = finished
        }
    }
    return (text, calls, try #require(completion))
}

/// GLM 5.3 Flash through Local Code's own chat format. Set the library and
/// the GLM directory to enable it:
///
///     ENGINEROUTER_LOCAL_CODE_LIBRARY=/path/to/PicoLocalCodeNative.dylib \
///     ENGINEROUTER_LOCAL_CODE_GLM_DIR=/path/to/glm swift test --filter LocalCodeGLMLiveTests
@Suite(.enabled(if: LiveLocalCode.library != nil && LiveLocalCode.glmDirectory != nil), .serialized)
struct LocalCodeGLMLiveTests {
    @Test(.timeLimit(.minutes(30)))
    func aToolRoundTripRunsThroughGLMsChatFormat() async throws {
        let engine = try LocalCodeEngine(libraryURL: URL(fileURLWithPath: try #require(LiveLocalCode.library)))
        let router = Router(engines: [engine], memoryBudget: Int(ProcessInfo.processInfo.physicalMemory))
        let model = EngineModel(
            id: "glm-5.3-flash", directory: URL(fileURLWithPath: try #require(LiveLocalCode.glmDirectory)))
        let loaded = try await router.open(model)
        let chat = try #require(loaded as? any ChatModel)
        // Its tokens are not DeepSeek's, so the token-level calls refuse it.
        await #expect(throws: LocalCodeError.chatOnly) { try await loaded.tokenize("x") }
        let weather = ChatTool(
            name: "get_weather", description: "Today's weather in a city.",
            parameters: #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#)
        var messages = [ChatMessage(role: .user, content: "What is the weather in Paris today? Use the tool.")]

        let first = try await chatTurn(chat, ChatRequest(
            messages: messages, tools: [weather], reasoning: false, maxTokens: 512, temperature: 0))
        print("GLM called: \(first.calls), after \(first.text.debugDescription)")
        let call = try #require(first.calls.first)
        #expect(call.name == "get_weather")
        #expect(call.arguments.contains("Paris"))
        #expect(first.completion.reason == .toolCalls)

        messages.append(ChatMessage(role: .assistant, content: first.text, toolCalls: first.calls))
        messages.append(ChatMessage(role: .tool, content: #"{"forecast":"sunny","celsius":21}"#, toolCallID: call.id))
        let request = ChatRequest(messages: messages, tools: [weather], reasoning: false, maxTokens: 512, temperature: 0)
        let counted = try await chat.countPromptTokens(request)
        let second = try await chatTurn(chat, request)
        print("GLM answered: \(second.text)")
        print("Second turn: \(second.completion.promptTokenCount) prompt tokens, \(second.completion.cachedPromptTokenCount) cached; counted \(counted)")
        #expect(second.text.contains("21") || second.text.localizedCaseInsensitiveContains("sunny"))
        #expect(second.completion.reason == .stop)
        #expect(second.completion.promptTokenCount == counted)
        #expect(second.completion.cachedPromptTokenCount > 0)

        await router.close(model.id)
    }
}

/// Runs against the real library and a DeepSeek 4.1 checkpoint. Set both
/// variables to enable it:
///
///     ENGINEROUTER_LOCAL_CODE_LIBRARY=/path/to/PicoLocalCodeNative.dylib \
///     ENGINEROUTER_LOCAL_CODE_MODEL_DIR=/path/to/deepseek swift test
@Suite(.enabled(if: LiveLocalCode.library != nil && LiveLocalCode.modelDirectory != nil), .serialized)
struct LocalCodeLiveTests {

    private static let question = "Reply in one short sentence: What is 7 plus 5?"
    private static let prompt = "<｜begin▁of▁sentence｜><｜User｜>\(question)<｜Assistant｜></think>"

    @Test(.timeLimit(.minutes(30)))
    func theRouterGeneratesThroughLocalCode() async throws {
        let engine = try LocalCodeEngine(libraryURL: URL(fileURLWithPath: try #require(LiveLocalCode.library)))
        let router = Router(engines: [engine], memoryBudget: Int(ProcessInfo.processInfo.physicalMemory))
        let model = EngineModel(
            id: "deepseek-4.1-flash", directory: URL(fileURLWithPath: try #require(LiveLocalCode.modelDirectory)))
        let loaded = try await router.open(model)
        let status = try #require(await router.status().first)
        #expect(status.engine == "local-code")
        #expect(status.loadBytes > 0)
        #expect(status.memory?.active == 0, "The weights load on the first generation")

        // Special tokens written in the text become single IDs; plain text round-trips.
        let prompt = try await loaded.tokenize(Self.prompt)
        #expect(prompt.first == 0)
        #expect(prompt.contains(128_803))
        #expect(try await loaded.detokenize(try await loaded.tokenize(Self.question)) == Self.question)
        let accented = "Crème brûlée, 日本語, 🙂"
        #expect(try await loaded.detokenize(try await loaded.tokenize(accented)) == accented)
        // A NUL character is text like any other.
        let withNUL = "a\u{0}b"
        #expect(try await loaded.detokenize(try await loaded.tokenize(withNUL)) == withNUL)

        // A greedy request runs to the model's stop token.
        let (tokens, completion) = try await run(
            loaded, EngineRequest(promptTokens: prompt, maxTokens: 64, temperature: 0))
        let answer = try await loaded.detokenize(tokens)
        print("Local Code answered: \(answer)")
        let finished = try #require(completion)
        #expect(answer.contains("12") || answer.localizedCaseInsensitiveContains("twelve"))
        #expect(finished.reason == .stop)
        #expect(finished.stopToken == 1, "DeepSeek's end-of-sentence token")
        #expect(finished.generatedTokenCount == tokens.count)
        #expect(finished.promptTokenCount == prompt.count)
        #expect(await loaded.memory().active == status.loadBytes)

        // The token limit ends a request early.
        let limited = try await run(loaded, EngineRequest(promptTokens: prompt, maxTokens: 2, temperature: 0))
        #expect(limited.completion?.reason == .length)
        #expect(limited.completion?.stopToken == nil)
        #expect(limited.completion?.generatedTokenCount == 2)
        #expect(limited.tokens == Array(tokens.prefix(2)))

        // Cancelling the consumer stops the request, and the model serves the next one.
        let consumer = Task {
            var received = 0
            for try await event in loaded.generate(EngineRequest(promptTokens: prompt, temperature: 0)) {
                if case .tokens(let ids) = event { received += ids.count }
                withUnsafeCurrentTask { $0?.cancel() }
            }
            return received
        }
        let received = (try? await consumer.value) ?? 0
        #expect(received < tokens.count)
        let afterCancel = try await run(loaded, EngineRequest(promptTokens: prompt, maxTokens: 2, temperature: 0))
        #expect(afterCancel.tokens == Array(tokens.prefix(2)))

        await router.close(model.id)
        #expect(await router.status().isEmpty)
        await #expect(throws: LocalCodeError.closed) {
            for try await _ in loaded.generate(EngineRequest(promptTokens: prompt)) {}
        }
    }

    /// A tool call, its result and the answer, laid out and read by Local
    /// Code's own chat format. The second turn goes on from the first one's
    /// reply, so Local Code takes its start from the first turn's KV cache.
    @Test(.timeLimit(.minutes(30)))
    func aToolRoundTripRunsThroughLocalCodesChatFormat() async throws {
        let engine = try LocalCodeEngine(libraryURL: URL(fileURLWithPath: try #require(LiveLocalCode.library)))
        let router = Router(engines: [engine], memoryBudget: Int(ProcessInfo.processInfo.physicalMemory))
        let model = EngineModel(
            id: "deepseek-4.1-flash", directory: URL(fileURLWithPath: try #require(LiveLocalCode.modelDirectory)))
        let chat = try #require(try await router.open(model) as? any ChatModel)
        let weather = ChatTool(
            name: "get_weather", description: "Today's weather in a city.",
            parameters: #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#)
        var messages = [ChatMessage(role: .user, content: "What is the weather in Paris today? Use the tool.")]

        let first = try await chatTurn(chat, ChatRequest(
            messages: messages, tools: [weather], reasoning: false, maxTokens: 512, temperature: 0))
        print("Local Code called: \(first.calls), after \(first.text.debugDescription)")
        let call = try #require(first.calls.first)
        #expect(call.name == "get_weather")
        #expect(call.arguments.contains("Paris"))
        #expect(first.completion.reason == .toolCalls)

        messages.append(ChatMessage(role: .assistant, content: first.text, toolCalls: first.calls))
        messages.append(ChatMessage(role: .tool, content: #"{"forecast":"sunny","celsius":21}"#, toolCallID: call.id))
        let request = ChatRequest(messages: messages, tools: [weather], reasoning: false, maxTokens: 512, temperature: 0)
        let counted = try await chat.countPromptTokens(request)
        let second = try await chatTurn(chat, request)
        print("Local Code answered: \(second.text)")
        print("Second turn: \(second.completion.promptTokenCount) prompt tokens, \(second.completion.cachedPromptTokenCount) cached; counted \(counted)")
        #expect(second.text.contains("21") || second.text.localizedCaseInsensitiveContains("sunny"))
        #expect(second.completion.reason == .stop)
        #expect(second.completion.promptTokenCount == counted)
        #expect(second.completion.cachedPromptTokenCount > 0)

        await router.close(model.id)
    }

    @Test(.timeLimit(.minutes(30)))
    func localCodeMeetsTheEngineContract() async throws {
        let engine = try LocalCodeEngine(libraryURL: URL(fileURLWithPath: try #require(LiveLocalCode.library)))
        let model = EngineModel(
            id: "deepseek-4.1-flash", directory: URL(fileURLWithPath: try #require(LiveLocalCode.modelDirectory)))

        #expect(await engineContractViolations(of: engine, model: model, prompt: Self.question) == [])
    }

    private func run(
        _ model: any LoadedModel, _ request: EngineRequest
    ) async throws -> (tokens: [Int], completion: EngineCompletion?) {
        var tokens: [Int] = []
        var completion: EngineCompletion?
        for try await event in model.generate(request) {
            switch event {
            case .tokens(let ids): tokens += ids
            case .finished(let finished): completion = finished
            }
        }
        return (tokens, completion)
    }
}
#endif
