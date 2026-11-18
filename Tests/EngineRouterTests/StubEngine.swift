import EngineRouter
import Foundation
import Synchronization

/// An engine with no model behind it. It serves model IDs that start with its
/// name and "generates" a fixed token sequence.
struct StubEngine: InferenceEngine {
    let name: String
    var isExclusive = false
    var loadBytes = 100
    var failsToOpen = false
    /// Tokens every request yields, one event per token.
    var tokens: [Int] = [1, 2, 3]
    /// When set, a request waits here after its first token until cancelled.
    var stallsAfterFirstToken = false
    /// When set, `close` waits here before the model lets go.
    var closeGate: Gate?
    /// Faults for the contract tests to catch.
    var ignoresMaxTokens = false
    var servesAfterClose = false
    var failsToTokenize = false
    var namesAStopTokenAtTheLimit = false
    let log = Log()

    func canServe(_ model: EngineModel) async -> Bool { model.id.hasPrefix(name) }

    func loadBytes(_ model: EngineModel) async throws -> Int { loadBytes }

    func open(_ model: EngineModel) async throws -> any LoadedModel {
        if failsToOpen { throw StubError.cannotOpen }
        return StubModel(engine: self, id: model.id)
    }

    /// What happened to the engine's models, for tests to observe.
    final class Log: Sendable {
        private let state = Mutex<(closed: [String], cancelled: Int, trimmed: Int)>(([], 0, 0))
        private let cancellations: AsyncStream<Void>
        private let cancellationSignal: AsyncStream<Void>.Continuation
        private let closeStarts: AsyncStream<Void>
        private let closeStartSignal: AsyncStream<Void>.Continuation

        init() {
            (cancellations, cancellationSignal) = AsyncStream.makeStream()
            (closeStarts, closeStartSignal) = AsyncStream.makeStream()
        }

        var closed: [String] { state.withLock { $0.closed } }
        var trimmed: Int { state.withLock { $0.trimmed } }

        func close(_ id: String) { state.withLock { $0.closed.append(id) } }
        func trim(_ bytes: Int) { state.withLock { $0.trimmed += bytes } }
        func cancel() {
            state.withLock { $0.cancelled += 1 }
            cancellationSignal.yield()
        }

        /// Returns once a request has been cancelled.
        func waitForCancellation() async {
            for await _ in cancellations { return }
        }

        func closeStarted() { closeStartSignal.yield() }

        /// Returns once a model has entered `close`.
        func waitForCloseStart() async {
            for await _ in closeStarts { return }
        }
    }

    /// Holds callers of `wait` until `open` is called.
    final class Gate: Sendable {
        private let state = Mutex<(isOpen: Bool, waiters: [CheckedContinuation<Void, Never>])>((false, []))

        func wait() async {
            await withCheckedContinuation { continuation in
                let isOpen = state.withLock { state in
                    if !state.isOpen { state.waiters.append(continuation) }
                    return state.isOpen
                }
                if isOpen { continuation.resume() }
            }
        }

        func open() {
            let waiters = state.withLock { state in
                state.isOpen = true
                defer { state.waiters = [] }
                return state.waiters
            }
            for waiter in waiters { waiter.resume() }
        }
    }
}

enum StubError: Error { case cannotOpen, cannotTokenize, closed }

final class StubModel: LoadedModel {
    let engine: StubEngine
    let id: String
    /// This instance's own state: the same ID can be opened again after a close.
    private let isClosed = Mutex(false)

    init(engine: StubEngine, id: String) {
        self.engine = engine
        self.id = id
    }

    func tokenize(_ text: String) async throws -> [Int] {
        if engine.failsToTokenize { throw StubError.cannotTokenize }
        return text.utf8.map(Int.init)
    }

    func detokenize(_ tokens: [Int]) async throws -> String {
        String(decoding: tokens.map(UInt8.init), as: UTF8.self)
    }

    func generate(_ request: EngineRequest) -> AsyncThrowingStream<EngineEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream<EngineEvent, any Error>.makeStream()
        if isClosed.withLock({ $0 }), !engine.servesAfterClose {
            continuation.finish(throwing: StubError.closed)
            return stream
        }
        let produced = Array(engine.tokens.prefix(engine.ignoresMaxTokens ? .max : request.maxTokens ?? .max))
        let stalls = engine.stallsAfterFirstToken
        let log = engine.log
        let task = Task {
            for (index, token) in produced.enumerated() {
                continuation.yield(.tokens([token]))
                if stalls, index == 0 {
                    // Sleeps until the consumer goes away.
                    try? await Task.sleep(for: .seconds(3600))
                    return
                }
            }
            let stopped = produced.count == engine.tokens.count
            continuation.yield(.finished(EngineCompletion(
                reason: stopped ? .stop : .length,
                promptTokenCount: request.promptTokens.count, generatedTokenCount: produced.count,
                promptTime: 0, generationTime: 0,
                stopToken: stopped || engine.namesAStopTokenAtTheLimit ? 0 : nil)))
            continuation.finish()
        }
        continuation.onTermination = { termination in
            if case .cancelled = termination {
                task.cancel()
                log.cancel()
            }
        }
        return stream
    }

    func memory() async -> EngineMemory {
        EngineMemory(active: engine.loadBytes, cache: 10, peak: engine.loadBytes)
    }

    func trim(toFree bytes: Int) async -> Int {
        let freed = min(bytes, 10)
        engine.log.trim(freed)
        return freed
    }

    func close() async {
        engine.log.closeStarted()
        await engine.closeGate?.wait()
        isClosed.withLock { $0 = true }
        engine.log.close(id)
    }
}
