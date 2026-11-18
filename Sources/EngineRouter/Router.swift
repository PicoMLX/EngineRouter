/// Routes each model to the first engine that can serve it and keeps every
/// open model inside one memory budget.
///
/// The router never closes a model on its own. When a model does not fit, or
/// an exclusive engine needs the memory to itself, `open` throws and the
/// caller decides what to close.
public actor Router {
    private struct Entry {
        let engine: String
        let isExclusive: Bool
        let loadBytes: Int
        let opening: Task<any LoadedModel, any Error>
        /// Set once `opening` has succeeded.
        var model: (any LoadedModel)?
        /// Set while `close` is running. The entry stays until it finishes, so
        /// the model keeps its bytes and its exclusivity until it has let go.
        var closing: Task<Void, Never>?
    }

    private let engines: [any InferenceEngine]
    private let memoryBudget: Int
    private var entries: [String: Entry] = [:]

    /// - Parameters:
    ///   - engines: In priority order. A model goes to the first one whose
    ///     `canServe` returns true.
    ///   - memoryBudget: The bytes all open models may hold together.
    public init(engines: [any InferenceEngine], memoryBudget: Int) {
        self.engines = engines
        self.memoryBudget = memoryBudget
    }

    /// Known build information by engine name, available before any models open.
    /// When a name is registered more than once, the first registration supplies it.
    public nonisolated func engineBuildInfos() -> [String: EngineBuildInfo] {
        Dictionary(engines.map { ($0.name, $0.buildInfo) }, uniquingKeysWith: { first, _ in first })
            .compactMapValues { $0 }
    }

    /// Opens `model`, or returns it when it is already open or opening. A
    /// model that is closing finishes closing first and then opens afresh.
    public func open(_ model: EngineModel) async throws -> any LoadedModel {
        if let loaded = try await existing(model.id) { return loaded }

        var serving: (any InferenceEngine)?
        for engine in engines where await engine.canServe(model) {
            serving = engine
            break
        }
        guard let engine = serving else { throw RouterError.noEngine(model: model.id) }
        let bytes = try await engine.loadBytes(model)

        // Another caller may have opened this model during the awaits above.
        if let loaded = try await existing(model.id) { return loaded }
        try admit(engine, loadBytes: bytes)

        // No suspension between `admit` and this insert, so two openers cannot
        // both take the same free budget.
        let opening = Task { try await engine.open(model) }
        entries[model.id] = Entry(
            engine: engine.name, isExclusive: engine.isExclusive, loadBytes: bytes, opening: opening)
        do {
            let loaded = try await opening.value
            if entries[model.id]?.opening == opening { entries[model.id]?.model = loaded }
            return loaded
        } catch {
            if entries[model.id]?.opening == opening { entries[model.id] = nil }
            throw error
        }
    }

    /// Closes the model, then returns its bytes to the budget. Does nothing
    /// for a model that is not open.
    public func close(_ modelID: String) async {
        guard let entry = entries[modelID] else { return }
        let closing: Task<Void, Never>
        if let running = entry.closing {
            closing = running
        } else {
            closing = Task { if let model = try? await entry.opening.value { await model.close() } }
            entries[modelID]?.closing = closing
        }
        await closing.value
        finishClosing(modelID, closing)
    }

    /// The model under `id` when it is open or opening, after waiting out a
    /// close in progress; `nil` once nothing holds the ID.
    private func existing(_ id: String) async throws -> (any LoadedModel)? {
        while let entry = entries[id] {
            guard let closing = entry.closing else { return try await entry.opening.value }
            await closing.value
            // Whoever resumes first removes the entry, so this loop ends.
            finishClosing(id, closing)
        }
        return nil
    }

    private func finishClosing(_ id: String, _ closing: Task<Void, Never>) {
        if entries[id]?.closing == closing { entries[id] = nil }
    }

    /// Every model that holds budget, sorted by ID: open, opening or closing.
    public func status() async -> [ModelStatus] {
        var result: [ModelStatus] = []
        for (id, entry) in entries.sorted(by: { $0.key < $1.key }) {
            result.append(ModelStatus(
                model: id, engine: entry.engine, loadBytes: entry.loadBytes, memory: await entry.model?.memory()))
        }
        return result
    }

    /// Asks open models to drop caches until about `bytes` are free. Returns
    /// the bytes freed.
    public func trim(toFree bytes: Int) async -> Int {
        var freed = 0
        for entry in entries.values {
            guard freed < bytes, entry.closing == nil, let model = entry.model else { continue }
            freed += await model.trim(toFree: bytes - freed)
        }
        return freed
    }

    private func admit(_ engine: any InferenceEngine, loadBytes: Int) throws {
        if !entries.isEmpty, engine.isExclusive || entries.values.contains(where: \.isExclusive) {
            throw RouterError.exclusiveConflict(openModels: entries.keys.sorted())
        }
        let available = memoryBudget - entries.values.reduce(0) { $0 + $1.loadBytes }
        guard loadBytes <= available else {
            throw RouterError.insufficientMemory(needed: loadBytes, available: available)
        }
    }
}

public struct ModelStatus: Sendable, Equatable {
    public var model: String
    public var engine: String
    /// The bytes charged against the router's budget.
    public var loadBytes: Int
    /// The model's own report; `nil` while it is still opening.
    public var memory: EngineMemory?

    public init(model: String, engine: String, loadBytes: Int, memory: EngineMemory?) {
        self.model = model
        self.engine = engine
        self.loadBytes = loadBytes
        self.memory = memory
    }
}

public enum RouterError: Error, Equatable, CustomStringConvertible {
    /// No registered engine can serve the model.
    case noEngine(model: String)
    /// The model's `loadBytes` exceed what the budget has left.
    case insufficientMemory(needed: Int, available: Int)
    /// The model, or one of `openModels`, belongs to an exclusive engine.
    case exclusiveConflict(openModels: [String])

    public var description: String {
        switch self {
        case .noEngine(let model):
            "No engine can serve \(model)."
        case .insufficientMemory(let needed, let available):
            "The model needs \(needed) bytes and the memory budget has \(available) left."
        case .exclusiveConflict(let openModels):
            "An engine that needs the machine's memory to itself cannot share it with: "
                + openModels.joined(separator: ", ") + "."
        }
    }
}
