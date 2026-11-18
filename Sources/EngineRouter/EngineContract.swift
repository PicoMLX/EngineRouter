/// Runs `engine` through the behaviour every adapter must have and returns one
/// line per violation. An empty result means the engine conforms.
///
/// An adapter's tests call this with a real model. The model is opened, used
/// for a handful of short requests, and closed.
///
/// - Parameters:
///   - model: A model the engine is expected to serve.
///   - prompt: Plain text without special tokens. It must tokenize to at least
///     one token and lead the model to generate at least two.
public func engineContractViolations(
    of engine: any InferenceEngine, model: EngineModel, prompt: String
) async -> [String] {
    var violations: [String] = []
    func check(_ condition: Bool, _ violation: @autoclosure () -> String) {
        if !condition { violations.append(violation()) }
    }

    check(await engine.canServe(model), "canServe refuses the model.")
    let loaded: any LoadedModel
    do {
        check(try await engine.loadBytes(model) > 0, "loadBytes is not positive.")
        loaded = try await engine.open(model)
    } catch {
        return violations + ["Threw \(error)."]
    }

    // Set once the prompt has tokenized, for the check after `close`.
    var request: EngineRequest?
    do {
        let tokens = try await loaded.tokenize(prompt)
        check(!tokens.isEmpty, "tokenize returns no tokens for the prompt.")
        let roundTrip = try await loaded.detokenize(tokens)
        check(roundTrip == prompt, "detokenize(tokenize(prompt)) is \"\(roundTrip)\", not the prompt.")

        // A limited greedy request: tokens, then one completion, last.
        let limit = 2
        let limited = EngineRequest(promptTokens: tokens, maxTokens: limit, temperature: 0)
        request = limited
        let first = try await events(of: loaded.generate(limited))
        var generated: [Int] = []
        var completions: [EngineCompletion] = []
        for event in first {
            switch event {
            case .tokens(let ids):
                check(completions.isEmpty, "Tokens arrive after .finished.")
                check(!ids.isEmpty, "A .tokens event is empty.")
                generated += ids
            case .finished(let completion):
                completions.append(completion)
            }
        }
        check(completions.count == 1, "A finished request yields \(completions.count) completions, not 1.")
        check(generated.count <= limit, "\(generated.count) tokens exceed maxTokens \(limit).")
        if let completion = completions.first {
            check(
                completion.generatedTokenCount == generated.count,
                "The completion counts \(completion.generatedTokenCount) tokens; \(generated.count) were yielded.")
            check(
                completion.promptTokenCount + completion.cachedPromptTokenCount == tokens.count,
                "Evaluated plus cached prompt tokens do not add up to the prompt's \(tokens.count).")
            check(
                completion.reason != .length || generated.count == limit,
                "The reason is .length but only \(generated.count) of \(limit) tokens were generated.")
        }

        // Greedy generation repeats.
        let second = try await events(of: loaded.generate(limited))
        check(tokenIDs(second) == generated, "Two temperature-0 requests differ.")

        // A consumer that goes away cancels its request and leaves the model usable.
        let consumer = Task {
            for try await _ in loaded.generate(EngineRequest(promptTokens: tokens, temperature: 0)) {
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        _ = try? await consumer.value
        let afterCancel = try await events(of: loaded.generate(limited))
        check(tokenIDs(afterCancel) == generated, "The request after a cancelled one differs.")

        let memory = await loaded.memory()
        check(memory.active >= 0 && memory.cache >= 0, "memory reports negative bytes.")
        check(memory.peak >= memory.active, "memory.peak is below memory.active.")
    } catch {
        violations.append("Threw \(error).")
    }

    // The model is closed however the checks above ended.
    await loaded.close()
    if let request {
        let afterClose = try? await events(of: loaded.generate(request))
        check(afterClose == nil, "generate on a closed model does not throw.")
    }
    return violations
}

private func events(of stream: AsyncThrowingStream<EngineEvent, any Error>) async throws -> [EngineEvent] {
    var result: [EngineEvent] = []
    for try await event in stream { result.append(event) }
    return result
}

private func tokenIDs(_ events: [EngineEvent]) -> [Int] {
    events.flatMap { event -> [Int] in
        if case .tokens(let ids) = event { return ids }
        return []
    }
}
