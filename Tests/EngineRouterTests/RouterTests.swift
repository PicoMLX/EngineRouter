import EngineRouter
import Foundation
import Synchronization
import Testing

struct RouterTests {
    private func model(_ id: String) -> EngineModel {
        EngineModel(id: id, directory: URL(fileURLWithPath: "/models/\(id)"))
    }

    @Test
    func reportsRegisteredEngineBuildsWithoutOpeningAModel() {
        let first = EngineBuildInfo(revision: String(repeating: "a", count: 40), tag: "v1.2.3", dirty: false)
        let second = EngineBuildInfo(revision: String(repeating: "b", count: 40), dirty: true)
        let router = Router(
            engines: [
                StubEngine(name: "a", buildInfo: first), StubEngine(name: "a", buildInfo: second),
                StubEngine(name: "b", buildInfo: second), StubEngine(name: "unknown"),
            ],
            memoryBudget: 1000)

        #expect(router.engineBuildInfos() == ["a": first, "b": second])
    }

    @Test
    func aModelGoesToTheFirstEngineThatCanServeIt() async throws {
        let router = Router(
            engines: [StubEngine(name: "a"), StubEngine(name: "b"), StubEngine(name: "b", loadBytes: 1)],
            memoryBudget: 1000)

        _ = try await router.open(model("b-small"))

        #expect(await router.status() == [
            ModelStatus(
                model: "b-small", engine: "b", loadBytes: 100,
                memory: EngineMemory(active: 100, cache: 10, peak: 100))
        ])
    }

    @Test
    func aModelNoEngineServesIsRefused() async {
        let router = Router(engines: [StubEngine(name: "a")], memoryBudget: 1000)

        await #expect(throws: RouterError.noEngine(model: "z-model")) {
            try await router.open(model("z-model"))
        }
    }

    @Test
    func openingAnOpenModelChargesTheBudgetOnce() async throws {
        let router = Router(engines: [StubEngine(name: "a")], memoryBudget: 250)

        _ = try await router.open(model("a-one"))
        _ = try await router.open(model("a-one"))
        // 250 holds two models of 100 bytes, so this fits only if "a-one" was charged once.
        _ = try await router.open(model("a-two"))

        #expect(await router.status().map(\.model) == ["a-one", "a-two"])
    }

    @Test
    func concurrentOpenersCannotOverfillTheBudget() async {
        let router = Router(engines: [StubEngine(name: "a")], memoryBudget: 250)

        let opened = await withTaskGroup(of: Bool.self) { group in
            for index in 0..<8 {
                group.addTask { (try? await router.open(self.model("a-\(index)"))) != nil }
            }
            return await group.reduce(0) { $0 + ($1 ? 1 : 0) }
        }

        #expect(opened == 2)
        #expect(await router.status().count == 2)
    }

    @Test
    func aModelThatDoesNotFitIsRefusedUntilAnotherCloses() async throws {
        let engine = StubEngine(name: "a")
        let router = Router(engines: [engine], memoryBudget: 150)
        _ = try await router.open(model("a-one"))

        await #expect(throws: RouterError.insufficientMemory(needed: 100, available: 50)) {
            try await router.open(model("a-two"))
        }

        await router.close("a-one")
        _ = try await router.open(model("a-two"))
        #expect(engine.log.closed == ["a-one"])
        #expect(await router.status().map(\.model) == ["a-two"])
    }

    @Test(.timeLimit(.minutes(1)))
    func aClosingModelKeepsItsBytesUntilItHasClosed() async throws {
        let engine = StubEngine(name: "a", closeGate: StubEngine.Gate())
        let router = Router(engines: [engine], memoryBudget: 150)
        _ = try await router.open(model("a-one"))
        let closing = Task { await router.close("a-one") }
        await engine.log.waitForCloseStart()

        await #expect(throws: RouterError.insufficientMemory(needed: 100, available: 50)) {
            try await router.open(model("a-two"))
        }

        engine.closeGate?.open()
        await closing.value
        _ = try await router.open(model("a-two"))
        #expect(await router.status().map(\.model) == ["a-two"])
    }

    @Test(.timeLimit(.minutes(1)))
    func aClosingExclusiveModelStillKeepsOthersOut() async throws {
        let solo = StubEngine(name: "solo", isExclusive: true, closeGate: StubEngine.Gate())
        let router = Router(engines: [StubEngine(name: "a"), solo], memoryBudget: 1000)
        _ = try await router.open(model("solo-one"))
        let closing = Task { await router.close("solo-one") }
        await solo.log.waitForCloseStart()

        await #expect(throws: RouterError.exclusiveConflict(openModels: ["solo-one"])) {
            try await router.open(model("a-one"))
        }

        solo.closeGate?.open()
        await closing.value
        _ = try await router.open(model("a-one"))
    }

    @Test(.timeLimit(.minutes(1)))
    func reopeningAClosingModelWaitsForTheCloseThenOpensAfresh() async throws {
        let engine = StubEngine(name: "a", closeGate: StubEngine.Gate())
        let router = Router(engines: [engine], memoryBudget: 100)
        _ = try await router.open(model("a-one"))
        let closing = Task { await router.close("a-one") }
        await engine.log.waitForCloseStart()

        let reopened = Mutex(false)
        let reopening = Task {
            _ = try await router.open(self.model("a-one"))
            reopened.withLock { $0 = true }
        }
        // The reopen cannot finish while the close is held.
        try await Task.sleep(for: .milliseconds(50))
        #expect(!reopened.withLock { $0 })
        engine.closeGate?.open()
        try await reopening.value
        await closing.value

        #expect(engine.log.closed == ["a-one"])
        #expect(await router.status() == [
            ModelStatus(
                model: "a-one", engine: "a", loadBytes: 100,
                memory: EngineMemory(active: 100, cache: 10, peak: 100))
        ])
    }

    @Test
    func aFailedOpenReturnsItsBytes() async throws {
        let router = Router(
            engines: [StubEngine(name: "broken", failsToOpen: true), StubEngine(name: "a")],
            memoryBudget: 100)

        await #expect(throws: StubError.cannotOpen) { try await router.open(model("broken-one")) }

        _ = try await router.open(model("a-one"))
        #expect(await router.status().map(\.model) == ["a-one"])
    }

    @Test
    func anExclusiveEngineOpensOnlyIntoAnEmptyRouter() async throws {
        let router = Router(
            engines: [StubEngine(name: "a"), StubEngine(name: "solo", isExclusive: true)],
            memoryBudget: 1000)
        _ = try await router.open(model("a-one"))

        await #expect(throws: RouterError.exclusiveConflict(openModels: ["a-one"])) {
            try await router.open(model("solo-one"))
        }

        await router.close("a-one")
        _ = try await router.open(model("solo-one"))
        #expect(await router.status().map(\.engine) == ["solo"])
    }

    @Test
    func nothingOpensBesideAnExclusiveModel() async throws {
        let router = Router(
            engines: [StubEngine(name: "a"), StubEngine(name: "solo", isExclusive: true)],
            memoryBudget: 1000)
        _ = try await router.open(model("solo-one"))

        await #expect(throws: RouterError.exclusiveConflict(openModels: ["solo-one"])) {
            try await router.open(model("a-one"))
        }
    }

    @Test
    func trimStopsOnceEnoughIsFreed() async throws {
        let engine = StubEngine(name: "a")
        let router = Router(engines: [engine], memoryBudget: 1000)
        for id in ["a-one", "a-two", "a-three"] { _ = try await router.open(model(id)) }

        // Each stub model can free 10 bytes.
        #expect(await router.trim(toFree: 15) == 15)
        #expect(engine.log.trimmed == 15)
        #expect(await router.trim(toFree: 100) == 30)
    }

    @Test
    func aRequestStreamsItsTokensThenItsCompletion() async throws {
        let router = Router(engines: [StubEngine(name: "a", tokens: [7, 8, 9])], memoryBudget: 1000)
        let loaded = try await router.open(model("a-one"))
        let prompt = try await loaded.tokenize("hi")

        var events: [EngineEvent] = []
        for try await event in loaded.generate(EngineRequest(promptTokens: prompt, maxTokens: 2)) {
            events.append(event)
        }

        #expect(events == [
            .tokens([7]), .tokens([8]),
            .finished(EngineCompletion(
                reason: .length, promptTokenCount: 2, generatedTokenCount: 2, promptTime: 0, generationTime: 0)),
        ])
        #expect(try await loaded.detokenize(prompt) == "hi")
    }

    @Test(.timeLimit(.minutes(1)))
    func cancellingTheConsumerCancelsTheRequest() async throws {
        let engine = StubEngine(name: "a", stallsAfterFirstToken: true)
        let router = Router(engines: [engine], memoryBudget: 1000)
        let loaded = try await router.open(model("a-one"))

        let consumer = Task {
            var received = 0
            for try await _ in loaded.generate(EngineRequest(promptTokens: [1])) { received += 1 }
            return received
        }
        consumer.cancel()
        await engine.log.waitForCancellation()

        // The stalled request never finishes on its own, so returning at all
        // shows the cancellation reached the engine.
        #expect(try await consumer.value <= 1)
    }
}
