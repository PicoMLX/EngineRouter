import EngineRouter
import Foundation
import Testing

struct EngineContractTests {
    private let model = EngineModel(id: "a-one", directory: URL(fileURLWithPath: "/models/a-one"))

    @Test
    func aConformingEngineHasNoViolations() async {
        #expect(await engineContractViolations(of: StubEngine(name: "a"), model: model, prompt: "hi") == [])
    }

    @Test
    func anEngineThatIgnoresTheTokenLimitIsCaught() async {
        let violations = await engineContractViolations(
            of: StubEngine(name: "a", ignoresMaxTokens: true), model: model, prompt: "hi")

        #expect(violations == ["3 tokens exceed maxTokens 2."])
    }

    @Test
    func anEngineThatServesAClosedModelIsCaught() async {
        let violations = await engineContractViolations(
            of: StubEngine(name: "a", servesAfterClose: true), model: model, prompt: "hi")

        #expect(violations == ["generate on a closed model does not throw."])
    }

    @Test
    func anEngineThatCannotServeTheModelIsCaught() async {
        let violations = await engineContractViolations(of: StubEngine(name: "b"), model: model, prompt: "hi")

        #expect(violations.first == "canServe refuses the model.")
    }

    @Test
    func aModelIsClosedWhenACheckThrows() async {
        let engine = StubEngine(name: "a", failsToTokenize: true)

        let violations = await engineContractViolations(of: engine, model: model, prompt: "hi")

        #expect(violations == ["Threw cannotTokenize."])
        #expect(engine.log.closed == ["a-one"])
    }

    @Test
    func theSameEngineAndModelCanBeCheckedTwice() async {
        let engine = StubEngine(name: "a")

        #expect(await engineContractViolations(of: engine, model: model, prompt: "hi") == [])
        #expect(await engineContractViolations(of: engine, model: model, prompt: "hi") == [])
    }
}
