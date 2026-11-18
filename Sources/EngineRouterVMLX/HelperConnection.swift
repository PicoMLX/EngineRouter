#if os(macOS)
import Foundation
import Synchronization

/// A message from the helper. Which fields are set depends on what it answers.
struct HelperReply: Decodable, Sendable {
    struct Result: Decodable, Sendable {
        var `protocol`: Int?
        var engine: String?
        var canServe: Bool?
        var loadBytes: Int?
        var model: Int?
        var tokens: [Int]?
        var text: String?
        var active: Int?
        var cache: Int?
        var peak: Int?
        var freed: Int?
    }

    var id: UInt64
    /// The answer to a request that has one answer.
    var result: Result?
    var error: String?
    /// For a generation: `tokens`, `finished` or `cancelled`.
    var event: String?
    var tokens: [Int]?
    var reason: String?
    var promptTokens: Int?
    var cachedPromptTokens: Int?
    var generatedTokens: Int?
    var promptSeconds: Double?
    var generationSeconds: Double?
    var stopToken: Int?
}

/// The byte stream to a helper process and the requests in flight on it.
///
/// A message in either direction is a 4-byte big-endian length followed by
/// that many bytes of JSON. Every request carries an ID; the helper's answers
/// carry the same one. A generation answers with any number of `tokens`
/// events and then one message that ends it.
final class HelperConnection: Sendable {
    /// The version of the message protocol this adapter speaks.
    static let protocolVersion = 1
    static let maximumFrameBytes = 64 << 20

    private enum Waiter {
        case reply(CheckedContinuation<HelperReply, any Error>)
        case stream(AsyncThrowingStream<HelperReply, any Error>.Continuation)
    }

    private struct State {
        var nextID: UInt64 = 1
        var waiters: [UInt64: Waiter] = [:]
        /// Set once the helper's side of the stream has ended.
        var isClosed = false
    }

    private let toHelper: FileHandle
    private let state = Mutex(State())
    /// Whole frames are written one at a time.
    private let writing = Mutex(())

    /// - Parameters:
    ///   - toHelper: Where requests are written.
    ///   - fromHelper: Where the helper's messages are read, on a thread of
    ///     the connection's own until the stream ends.
    init(toHelper: FileHandle, fromHelper: FileHandle) {
        self.toHelper = toHelper
        let reader = Thread { [self] in
            while let frame = Self.readFrame(from: fromHelper) {
                guard let reply = try? JSONDecoder().decode(HelperReply.self, from: frame) else { continue }
                deliver(reply)
            }
            end()
        }
        reader.name = "enginerouter.vmlx.reader"
        reader.start()
    }

    /// Sends a request that has one answer and returns its result.
    func call(_ op: String, _ parameters: [String: Any] = [:]) async throws -> HelperReply.Result {
        let frame = try Self.frame(parameters)
        let reply = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<HelperReply, any Error>) in
            let id = register(.reply(continuation))
            guard let id else { return continuation.resume(throwing: VMLXError.helperExited) }
            if !send(frame(id, op)) { fail(id) }
        }
        if let message = reply.error { throw VMLXError.helper(message) }
        return reply.result ?? HelperReply.Result()
    }

    /// Starts a generation. The stream yields the helper's messages for it and
    /// finishes after the one that ends it; `id` is what `cancel` names.
    func stream(
        _ op: String, _ parameters: [String: Any]
    ) throws -> (id: UInt64, replies: AsyncThrowingStream<HelperReply, any Error>) {
        let frame = try Self.frame(parameters)
        let (replies, continuation) = AsyncThrowingStream<HelperReply, any Error>.makeStream()
        guard let id = register(.stream(continuation)) else { throw VMLXError.helperExited }
        if !send(frame(id, op)) { fail(id) }
        return (id, replies)
    }

    /// Sends a request the helper does not answer.
    func notify(_ op: String, _ parameters: [String: Any]) {
        guard let frame = try? Self.frame(parameters) else { return }
        _ = send(frame(0, op))
    }

    /// Ends the stream to the helper, which makes it close its models and exit.
    func close() {
        try? toHelper.close()
    }

    // MARK: - Requests in flight

    private func register(_ waiter: Waiter) -> UInt64? {
        state.withLock { state in
            guard !state.isClosed else { return nil }
            let id = state.nextID
            state.nextID += 1
            state.waiters[id] = waiter
            return id
        }
    }

    private func deliver(_ reply: HelperReply) {
        let waiter: Waiter? = state.withLock { state in
            guard let waiter = state.waiters[reply.id] else { return nil }
            // A `tokens` event is the only message that leaves a request in flight.
            if case .stream = waiter, reply.event == "tokens" { return waiter }
            state.waiters[reply.id] = nil
            return waiter
        }
        switch waiter {
        case .reply(let continuation):
            continuation.resume(returning: reply)
        case .stream(let continuation):
            continuation.yield(reply)
            if reply.event != "tokens" { continuation.finish() }
        case nil:
            break
        }
    }

    private func fail(_ id: UInt64) {
        finish(state.withLock { $0.waiters.removeValue(forKey: id) }.map { [$0] } ?? [])
    }

    /// The helper is gone: every request in flight fails, and so does every later one.
    private func end() {
        finish(state.withLock { state in
            state.isClosed = true
            defer { state.waiters = [:] }
            return Array(state.waiters.values)
        })
    }

    private func finish(_ waiters: [Waiter]) {
        for waiter in waiters {
            switch waiter {
            case .reply(let continuation): continuation.resume(throwing: VMLXError.helperExited)
            case .stream(let continuation): continuation.finish(throwing: VMLXError.helperExited)
            }
        }
    }

    // MARK: - Frames

    /// A request's JSON without its ID and operation, which `frame(id, op)` adds.
    private static func frame(_ parameters: [String: Any]) throws -> (UInt64, String) -> Data {
        guard JSONSerialization.isValidJSONObject(parameters) else { throw VMLXError.helper("A request is not JSON.") }
        return { id, op in
            var object = parameters
            object["id"] = id
            object["op"] = op
            // Checked above, and an ID and a name cannot make it invalid.
            let body = (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
            return Self.framed(body)
        }
    }

    static func framed(_ body: Data) -> Data {
        var frame = Data()
        withUnsafeBytes(of: UInt32(body.count).bigEndian) { frame.append(contentsOf: $0) }
        frame.append(body)
        return frame
    }

    private func send(_ frame: Data) -> Bool {
        writing.withLock { _ in (try? toHelper.write(contentsOf: frame)) != nil }
    }

    /// The next frame's body, or `nil` when the stream has ended or is not a frame.
    static func readFrame(from handle: FileHandle) -> Data? {
        guard let header = read(4, from: handle) else { return nil }
        let length = header.reduce(0) { $0 << 8 | Int($1) }
        guard length <= maximumFrameBytes else { return nil }
        return read(length, from: handle)
    }

    private static func read(_ count: Int, from handle: FileHandle) -> Data? {
        var data = Data()
        while data.count < count {
            guard let chunk = try? handle.read(upToCount: count - data.count), !chunk.isEmpty else { return nil }
            data.append(chunk)
        }
        return data
    }
}

public enum VMLXError: Error, Equatable, CustomStringConvertible {
    /// The helper could not be started or speaks another protocol version.
    case launch(String)
    /// The helper refused a request.
    case helper(String)
    /// The helper process ended. Its engine cannot be used again.
    case helperExited
    case closed

    public var description: String {
        switch self {
        case .launch(let message), .helper(let message): message
        case .helperExited: "The vmlx helper process has exited."
        case .closed: "The vmlx model is closed."
        }
    }
}
#endif
