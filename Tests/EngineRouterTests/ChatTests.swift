import EngineRouter
import Foundation
import Testing

struct ChatCompletionsBodyTests {
    private func body(_ request: ChatRequest) throws -> [String: Any] {
        let data = try request.chatCompletionsBody(model: "m")
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test
    func aConversationWithAToolRoundTripBecomesChatCompletionsMessages() throws {
        let call = ChatToolCall(id: "call_1", name: "weather", arguments: #"{"city":"Paris"}"#)
        let request = ChatRequest(
            messages: [
                ChatMessage(role: .system, content: "Be brief."),
                ChatMessage(role: .user, content: "Weather in Paris?"),
                ChatMessage(role: .assistant, content: "", reasoning: "Look it up.", toolCalls: [call]),
                ChatMessage(role: .tool, content: "Sunny", toolCallID: "call_1"),
            ],
            tools: [ChatTool(name: "weather", description: "Today's weather", parameters: #"{"type":"object"}"#)],
            toolChoice: .required, reasoning: false, maxTokens: 64, temperature: 0)

        let body = try body(request)

        let messages = try #require(body["messages"] as? [[String: Any]])
        #expect(messages.map { $0["role"] as? String } == ["system", "user", "assistant", "tool"])
        #expect(messages[2]["content"] is NSNull)
        #expect(messages[2]["reasoning_content"] as? String == "Look it up.")
        let toolCall = try #require((messages[2]["tool_calls"] as? [[String: Any]])?.first)
        #expect(toolCall["id"] as? String == "call_1")
        #expect((toolCall["function"] as? [String: Any])?["arguments"] as? String == #"{"city":"Paris"}"#)
        #expect(messages[3]["tool_call_id"] as? String == "call_1")
        let function = try #require((body["tools"] as? [[String: Any]])?.first?["function"] as? [String: Any])
        #expect(function["name"] as? String == "weather")
        #expect((function["parameters"] as? [String: Any])?["type"] as? String == "object")
        #expect(body["tool_choice"] as? String == "required")
        #expect(body["enable_thinking"] as? Bool == false)
        #expect(body["max_tokens"] as? Int == 64)
        #expect(body["stream"] as? Bool == true)
        #expect((body["stream_options"] as? [String: Any])?["include_usage"] as? Bool == true)
    }

    @Test
    func settingsLeftUnsetAreLeftOutForTheEngineToChoose() throws {
        let body = try body(ChatRequest(messages: [ChatMessage(role: .user, content: "Hi")]))

        for key in ["tools", "tool_choice", "enable_thinking", "max_tokens", "temperature", "top_p"] {
            #expect(body[key] == nil, "\(key)")
        }
    }

    @Test
    func toolParametersThatAreNotJSONAreRefused() {
        let request = ChatRequest(
            messages: [ChatMessage(role: .user, content: "Hi")], tools: [ChatTool(name: "t", parameters: "{")])

        #expect(throws: ChatError.self) { try request.chatCompletionsBody(model: "m") }
    }
}

struct ChatCompletionsStreamTests {
    private func events(_ chunks: [String]) throws -> (events: [ChatEvent], isFinished: Bool) {
        var stream = ChatCompletionsStream()
        var events: [ChatEvent] = []
        for chunk in chunks { events += try stream.events(fromChunk: Data(chunk.utf8)) }
        return (events, stream.isFinished)
    }

    private func delta(_ delta: String, finish: String? = nil) -> String {
        #"{"choices":[{"index":0,"delta":\#(delta),"finish_reason":\#(finish.map { "\"\($0)\"" } ?? "null")}]}"#
    }

    private let usage =
        #"{"choices":[],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":6},"#
        + #""completion_tokens":5,"completion_tokens_details":{"reasoning_tokens":2}}}"#

    @Test
    func reasoningAndTextArriveAsTheyStreamAndUsageEndsTheTurn() throws {
        let result = try events([
            delta(#"{"role":"assistant","content":""}"#),
            delta(#"{"reasoning_content":"Hm"}"#),
            delta(#"{"content":"Hello"}"#),
            delta("{}", finish: "stop"),
            usage,
        ])

        #expect(Array(result.events.dropLast()) == [.reasoning("Hm"), .text("Hello")])
        guard case .finished(let completion) = result.events.last else {
            Issue.record("The turn did not finish.")
            return
        }
        #expect(completion.reason == .stop)
        #expect(completion.promptTokenCount == 10)
        #expect(completion.cachedPromptTokenCount == 6)
        #expect(completion.generatedTokenCount == 5)
        #expect(completion.reasoningTokenCount == 2)
        #expect(result.isFinished)
    }

    @Test
    func aToolCallStreamedInPiecesArrivesWholeWhenTheTurnEnds() throws {
        let result = try events([
            delta(#"{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"weather","arguments":"{\"ci"}}]}"#),
            delta(#"{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]}"#),
            delta(#"{"tool_calls":[{"index":1,"function":{"name":"time"}}]}"#),
            delta("{}", finish: "tool_calls"),
            usage,
        ])

        #expect(Array(result.events.dropLast()) == [
            .toolCall(ChatToolCall(id: "call_a", name: "weather", arguments: #"{"city":"Paris"}"#)),
            .toolCall(ChatToolCall(id: "call_1", name: "time", arguments: "{}")),
        ])
        guard case .finished(let completion) = result.events.last else {
            Issue.record("The turn did not finish.")
            return
        }
        #expect(completion.reason == .toolCalls)
    }

    @Test
    func aTurnCutShortByItsLimitEndsWithLength() throws {
        let result = try events([delta(#"{"content":"Hel"}"#), delta("{}", finish: "length"), usage])

        guard case .finished(let completion) = result.events.last else {
            Issue.record("The turn did not finish.")
            return
        }
        #expect(completion.reason == .length)
    }

    /// The limit can stop the model inside a call, so no call from such a
    /// turn is handed over as complete.
    @Test
    func aTurnCutShortByItsLimitOffersNoneOfItsCalls() throws {
        let result = try events([
            delta(#"{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"weather","arguments":"{\"city\":"}}]}"#),
            delta("{}", finish: "length"),
            usage,
        ])

        #expect(result.events.count == 1)
        guard case .finished(let completion) = result.events.first else {
            Issue.record("The turn did not finish.")
            return
        }
        #expect(completion.reason == .length)
    }

    @Test
    func anErrorChunkThrowsItsMessage() {
        #expect(throws: ChatError.engine("No room.")) {
            _ = try events([#"{"error":{"code":"busy","message":"No room."}}"#])
        }
    }

    @Test
    func aTurnWithoutItsUsageChunkIsNotFinished() throws {
        let result = try events([delta(#"{"content":"Hi"}"#), delta("{}", finish: "stop")])

        #expect(result.events == [.text("Hi")])
        #expect(!result.isFinished)
    }
}
