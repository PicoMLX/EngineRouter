import Foundation

/// A model whose engine lays out its own prompts and reads its own replies.
///
/// The app sends the whole conversation and the tools it offers. The engine
/// renders them in the model's chat format, generates, and splits the reply
/// into reasoning, text and tool calls. It also keeps whatever caches let a
/// conversation's next turn skip the prompt it has already evaluated. The app
/// runs the tools and sends their results in the next request.
public protocol ChatModel: LoadedModel {
    /// Runs one turn of `request`'s conversation. A turn that runs to its end
    /// yields `.finished` last. Ending the stream early, by cancelling the
    /// consuming task or dropping the stream, cancels the turn.
    func chat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, any Error>

    /// The prompt tokens `request` holds once the engine lays it out, counting
    /// those a cache would supply.
    func countPromptTokens(_ request: ChatRequest) async throws -> Int
}

public struct ChatRequest: Sendable, Equatable {
    public var messages: [ChatMessage]
    public var tools: [ChatTool]
    public var toolChoice: ChatToolChoice
    /// Whether the model thinks before it answers; `nil` is the model's default.
    public var reasoning: Bool?
    /// The most tokens to generate; `nil` leaves the limit to the engine.
    public var maxTokens: Int?
    /// `0` is greedy; `nil` is the engine's default.
    public var temperature: Float?
    /// `nil` is the engine's default.
    public var topP: Float?

    public init(
        messages: [ChatMessage], tools: [ChatTool] = [], toolChoice: ChatToolChoice = .auto, reasoning: Bool? = nil,
        maxTokens: Int? = nil, temperature: Float? = nil, topP: Float? = nil
    ) {
        self.messages = messages
        self.tools = tools
        self.toolChoice = toolChoice
        self.reasoning = reasoning
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
    }
}

public struct ChatMessage: Sendable, Equatable {
    public enum Role: String, Sendable, Equatable {
        case system, user, assistant, tool
    }

    public var role: Role
    public var content: String
    /// An assistant message's thinking, which some chat formats keep.
    public var reasoning: String?
    /// The tools an assistant message called.
    public var toolCalls: [ChatToolCall]
    /// The call a tool message answers.
    public var toolCallID: String?

    public init(
        role: Role, content: String, reasoning: String? = nil, toolCalls: [ChatToolCall] = [],
        toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }
}

/// A function the model may call.
public struct ChatTool: Sendable, Equatable {
    public var name: String
    public var description: String?
    /// The JSON schema of the function's arguments, as JSON text.
    public var parameters: String?

    public init(name: String, description: String? = nil, parameters: String? = nil) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public enum ChatToolChoice: Sendable, Equatable {
    case auto
    case none
    /// The model must call one of the tools. Not every engine can hold a model to it.
    case required
    case function(String)
}

public struct ChatToolCall: Sendable, Equatable {
    public var id: String
    public var name: String
    /// The arguments, as JSON text.
    public var arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public enum ChatEvent: Sendable, Equatable {
    case reasoning(String)
    case text(String)
    /// A complete call, arriving once the model has finished writing it.
    case toolCall(ChatToolCall)
    case finished(ChatCompletion)
}

public struct ChatCompletion: Sendable, Equatable {
    public enum Reason: Sendable, Equatable {
        /// The model ended its turn.
        case stop
        /// The request's or the context's token limit was reached.
        case length
        /// The model ended its turn by calling tools.
        case toolCalls
    }

    public var reason: Reason
    /// Every prompt token, including those a cache supplied.
    public var promptTokenCount: Int
    /// Prompt tokens the engine reused from a cache instead of evaluating.
    public var cachedPromptTokenCount: Int
    /// Generated tokens, reasoning included.
    public var generatedTokenCount: Int
    /// Generated tokens of reasoning, when the engine reports them.
    public var reasoningTokenCount: Int?
    /// Seconds from submission to the first event.
    public var promptTime: TimeInterval
    /// Seconds from the first event to the last.
    public var generationTime: TimeInterval

    public init(
        reason: Reason, promptTokenCount: Int, cachedPromptTokenCount: Int = 0, generatedTokenCount: Int,
        reasoningTokenCount: Int? = nil, promptTime: TimeInterval, generationTime: TimeInterval
    ) {
        self.reason = reason
        self.promptTokenCount = promptTokenCount
        self.cachedPromptTokenCount = cachedPromptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.reasoningTokenCount = reasoningTokenCount
        self.promptTime = promptTime
        self.generationTime = generationTime
    }
}

public enum ChatError: Error, Equatable, CustomStringConvertible {
    /// The engine refused or failed the turn, for `message`.
    case engine(String)
    /// The engine sent something that is not a Chat Completions chunk.
    case malformedResponse
    /// The response ended before the turn's last chunk.
    case incomplete

    public var description: String {
        switch self {
        case .engine(let message): message
        case .malformedResponse: "The engine sent a reply this adapter can't read."
        case .incomplete: "The engine's reply ended early."
        }
    }
}

// MARK: - OpenAI Chat Completions, which the engines' chat entry points speak

package extension ChatRequest {
    /// The request as a Chat Completions body that streams, with usage at the end.
    func chatCompletionsBody(model: String) throws -> Data {
        var body: [String: Any] = [
            "model": model, "stream": true, "stream_options": ["include_usage": true],
            "messages": messages.map(\.chatCompletionsObject),
        ]
        if !tools.isEmpty {
            body["tools"] = try tools.map { try $0.chatCompletionsObject() }
        }
        switch toolChoice {
        case .auto: break
        case .none: body["tool_choice"] = "none"
        case .required: body["tool_choice"] = "required"
        case .function(let name): body["tool_choice"] = ["type": "function", "function": ["name": name]]
        }
        if let reasoning { body["enable_thinking"] = reasoning }
        if let maxTokens { body["max_tokens"] = maxTokens }
        if let temperature { body["temperature"] = Double(temperature) }
        if let topP { body["top_p"] = Double(topP) }
        return try JSONSerialization.data(withJSONObject: body)
    }
}

private extension ChatMessage {
    var chatCompletionsObject: [String: Any] {
        var object: [String: Any] = ["role": role.rawValue, "content": content]
        if let reasoning, !reasoning.isEmpty { object["reasoning_content"] = reasoning }
        if !toolCalls.isEmpty {
            object["tool_calls"] = toolCalls.map { call in
                ["id": call.id, "type": "function", "function": ["name": call.name, "arguments": call.arguments]]
            }
            // A turn of only tool calls has no text.
            if content.isEmpty { object["content"] = NSNull() }
        }
        if let toolCallID { object["tool_call_id"] = toolCallID }
        return object
    }
}

private extension ChatTool {
    func chatCompletionsObject() throws -> [String: Any] {
        var function: [String: Any] = ["name": name]
        if let description { function["description"] = description }
        if let parameters {
            guard let schema = try? JSONSerialization.jsonObject(with: Data(parameters.utf8)) else {
                throw ChatError.engine("The parameters of tool \(name) are not JSON.")
            }
            function["parameters"] = schema
        }
        return ["type": "function", "function": function]
    }
}

/// Reads a streamed Chat Completions response one chunk at a time, as chat
/// events. A tool call arrives whole, once the turn says it is over; its
/// pieces may come in any number of chunks before that.
package struct ChatCompletionsStream {
    private struct PendingCall {
        var id = ""
        var name = ""
        var arguments = ""
    }

    private let start: ContinuousClock.Instant
    private var firstEvent: ContinuousClock.Instant?
    private var calls: [Int: PendingCall] = [:]
    private var reason: ChatCompletion.Reason?
    private(set) package var isFinished = false

    package init(start: ContinuousClock.Instant = .now) {
        self.start = start
    }

    /// The events one chunk, a `data:` payload of the stream, brings.
    package mutating func events(fromChunk data: Data) throws -> [ChatEvent] {
        guard let chunk = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ChatError.malformedResponse
        }
        if let error = chunk["error"] as? [String: Any] {
            throw ChatError.engine(error["message"] as? String ?? "The engine failed the turn.")
        }
        firstEvent = firstEvent ?? .now
        var events: [ChatEvent] = []
        for choice in chunk["choices"] as? [[String: Any]] ?? [] {
            let delta = choice["delta"] as? [String: Any] ?? [:]
            if let text = delta["reasoning_content"] as? String ?? delta["reasoning"] as? String, !text.isEmpty {
                events.append(.reasoning(text))
            }
            if let text = delta["content"] as? String, !text.isEmpty {
                events.append(.text(text))
            }
            for piece in delta["tool_calls"] as? [[String: Any]] ?? [] {
                let index = (piece["index"] as? NSNumber)?.intValue ?? calls.count
                var call = calls[index] ?? PendingCall()
                if let id = piece["id"] as? String { call.id += id }
                let function = piece["function"] as? [String: Any]
                if let name = function?["name"] as? String { call.name += name }
                if let arguments = function?["arguments"] as? String { call.arguments += arguments }
                calls[index] = call
            }
            switch choice["finish_reason"] as? String {
            case nil: break
            case "length": reason = .length
            case "tool_calls": reason = .toolCalls
            default: reason = calls.isEmpty ? .stop : .toolCalls
            }
            if choice["finish_reason"] is String {
                // A turn its limit cut short may have stopped inside a call,
                // so none of its calls is handed over as complete.
                if reason != .length {
                    events += calls.keys.sorted().map { index in
                        let call = calls[index]!
                        return .toolCall(ChatToolCall(
                            id: call.id.isEmpty ? "call_\(index)" : call.id, name: call.name,
                            arguments: call.arguments.isEmpty ? "{}" : call.arguments))
                    }
                }
                calls = [:]
            }
        }
        if let usage = chunk["usage"] as? [String: Any], let reason {
            events.append(.finished(completion(reason: reason, usage: usage)))
            isFinished = true
        }
        return events
    }

    private func completion(reason: ChatCompletion.Reason, usage: [String: Any]) -> ChatCompletion {
        func count(_ object: Any?, _ key: String) -> Int? {
            ((object as? [String: Any])?[key] as? NSNumber)?.intValue
        }
        let end = ContinuousClock.now
        let first = firstEvent ?? end
        return ChatCompletion(
            reason: reason, promptTokenCount: count(usage, "prompt_tokens") ?? 0,
            cachedPromptTokenCount: count(usage["prompt_tokens_details"], "cached_tokens") ?? 0,
            generatedTokenCount: count(usage, "completion_tokens") ?? 0,
            reasoningTokenCount: count(usage["completion_tokens_details"], "reasoning_tokens"),
            promptTime: seconds(first - start), generationTime: seconds(end - first))
    }

    private func seconds(_ duration: Duration) -> TimeInterval {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}
