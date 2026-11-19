import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import VMLXTokenizers

let protocolVersion = 1
let engineDescription = "vmlx-swift, PicoMLX fork, engine-router"

/// Sequences decoded together per model by the batch engine.
let maxBatchSize = 8

/// Loads run one at a time across every server in the process: the library
/// adjusts process-wide MLX limits while loading and restores them afterwards.
let processLoads = LoadQueue()

actor LoadQueue {
    private var tail: Task<Void, Never>?

    /// Runs `load` once every load queued before it has finished.
    func run(_ load: @escaping @Sendable () async -> Void) async {
        let previous = tail
        let task = Task {
            await previous?.value
            await load()
        }
        tail = task
        await task.value
    }
}

// MARK: - Requests

struct Request: Decodable {
    let id: UInt64
    let op: String
    var directory: String?
    var model: Int?
    var text: String?
    var tokens: [Int]?
    var prompt: [Int]?
    var maxTokens: Int?
    var temperature: Double?
    var topP: Double?
    var request: UInt64?
    var bytes: Int?
}

private struct RequestID: Decodable { let id: UInt64 }

// MARK: - Model directories

struct ModelDirectoryFacts {
    let url: URL
    let modelType: String
    /// Size of the top-level `.safetensors` files, following symlinks.
    let weightBytes: Int
    let vocabularySize: Int?
    let contextLength: Int?
}

enum ModelDirectoryInspection {
    case model(ModelDirectoryFacts)
    case notModel(String)
}

/// Looks at a directory without loading weights: config, tokenizer files and
/// weight files must be present.
func inspectModelDirectory(_ path: String) -> ModelDirectoryInspection {
    guard path.hasPrefix("/") else { return .notModel("the directory must be an absolute path") }
    let fileManager = FileManager.default
    let url = URL(fileURLWithPath: path, isDirectory: true)
    let resolved = url.resolvingSymlinksInPath()
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: resolved.path, isDirectory: &isDirectory), isDirectory.boolValue
    else { return .notModel("no such directory") }

    guard let configData = try? Data(contentsOf: resolved.appendingPathComponent("config.json")),
        let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
    else { return .notModel("config.json is missing or unreadable") }
    guard let modelType = config["model_type"] as? String, !modelType.isEmpty else {
        return .notModel("config.json has no model_type")
    }
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        guard fileManager.fileExists(atPath: resolved.appendingPathComponent(name).path) else {
            return .notModel("\(name) is missing")
        }
    }

    var weightBytes = 0
    var weightFiles = 0
    for name in (try? fileManager.contentsOfDirectory(atPath: resolved.path)) ?? []
    where name.hasSuffix(".safetensors") {
        var info = stat()  // stat follows symlinks (Hugging Face snapshots link into a blob store)
        guard stat(resolved.appendingPathComponent(name).path, &info) == 0,
            info.st_mode & S_IFMT == S_IFREG
        else { continue }
        weightBytes += Int(info.st_size)
        weightFiles += 1
    }
    guard weightFiles > 0 else { return .notModel("no .safetensors weight files") }

    let text = config["text_config"] as? [String: Any]
    func integer(_ key: String) -> Int? { (text?[key] as? Int) ?? (config[key] as? Int) }
    return .model(
        ModelDirectoryFacts(
            url: url, modelType: modelType, weightBytes: weightBytes,
            vocabularySize: integer("vocab_size"), contextLength: integer("max_position_embeddings")))
}

func makeTokenizerLoader() -> any TokenizerLoader { #huggingFaceTokenizerLoader() }

// MARK: - Server state

/// One open model. Only touched from the `Server` actor.
final class Model {
    let handle: Int
    let facts: ModelDirectoryFacts
    var container: ModelContainer?
    var engine: BatchEngine?
    var tokenizer: (any MLXLMCommon.Tokenizer)?
    /// Temp-directory tokenizer shims the load wrote. The helper never loads a
    /// configuration twice, so it removes them when the model closes.
    var tokenizerShims: [URL] = []
    var closing = false
    var closeTask: Task<Void, Never>?
    /// In-flight generate requests on this model, by request id.
    var tasks: [UInt64: Task<Void, Never>] = [:]

    init(
        handle: Int, facts: ModelDirectoryFacts, container: ModelContainer, engine: BatchEngine,
        tokenizer: any MLXLMCommon.Tokenizer
    ) {
        self.handle = handle
        self.facts = facts
        self.container = container
        self.engine = engine
        self.tokenizer = tokenizer
    }
}

/// One in-flight generate request. Only touched from the `Server` actor.
final class Generation {
    let id: UInt64
    let model: Int
    var batchID: BatchRequestID?
    var cancelRequested = false

    init(id: UInt64, model: Int) {
        self.id = id
        self.model = model
    }
}

actor Server {
    private let out: FrameWriter
    private var models: [Int: Model] = [:]
    private var nextHandle = 1
    private var generations: [UInt64: Generation] = [:]
    /// This server's loads, in the order they arrived, so `shutdown` can wait for them.
    private var loadChain: Task<Void, Never>?
    private var knownModelTypes: Set<String>?

    init(out: FrameWriter) { self.out = out }

    // MARK: Responses

    private func reply(_ id: UInt64, _ result: [String: Any]) {
        out.send(["id": NSNumber(value: id), "result": result])
    }

    private func fail(_ id: UInt64, _ message: String) {
        out.send(["id": NSNumber(value: id), "error": message])
    }

    // MARK: Dispatch

    /// Handles one frame. Called in arrival order; anything slow is started
    /// here and finishes in its own task.
    func handle(_ data: Data) async {
        let request: Request
        do {
            request = try JSONDecoder().decode(Request.self, from: data)
        } catch {
            if let readable = try? JSONDecoder().decode(RequestID.self, from: data) {
                fail(readable.id, "malformed request: \(describe(error))")
            } else {
                log("dropping a frame with no readable id (\(data.count) bytes): \(describe(error))")
            }
            return
        }
        let id = request.id

        switch request.op {
        case "hello":
            reply(id, ["protocol": protocolVersion, "engine": engineDescription])

        case "canServe":
            guard let directory = request.directory else { return fail(id, "missing \"directory\"") }
            if case .model = await classify(directory) {
                reply(id, ["canServe": true])
            } else {
                reply(id, ["canServe": false])
            }

        case "loadBytes":
            guard let directory = request.directory else { return fail(id, "missing \"directory\"") }
            switch await classify(directory) {
            case .model(let facts): reply(id, ["loadBytes": facts.weightBytes])
            case .notModel(let why): fail(id, "not a loadable model directory: \(why)")
            }

        case "open":
            guard let directory = request.directory else { return fail(id, "missing \"directory\"") }
            let previous = loadChain
            loadChain = Task {
                await previous?.value
                await processLoads.run { await self.open(id: id, directory: directory) }
            }

        case "close":
            guard let model = openModel(request) else { return }
            let closing = closeTask(for: model)
            Task {
                await closing.value
                self.reply(id, [:])
            }

        case "tokenize":
            guard let model = openModel(request), let tokenizer = model.tokenizer else { return }
            guard let text = request.text else { return fail(id, "missing \"text\"") }
            reply(id, ["tokens": tokenizer.encode(text: text, addSpecialTokens: false)])

        case "detokenize":
            guard let model = openModel(request), let tokenizer = model.tokenizer else { return }
            guard let tokens = request.tokens else { return fail(id, "missing \"tokens\"") }
            if let bad = firstInvalidToken(tokens, model) {
                return fail(id, "token \(bad) is outside the model's vocabulary")
            }
            reply(id, ["text": tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)])

        case "generate":
            startGeneration(request)

        case "cancel":
            guard let target = request.request, let generation = generations[target] else { return }
            generation.cancelRequested = true
            if let batchID = generation.batchID, let engine = models[generation.model]?.engine {
                Task { await engine.cancel(batchID) }
            }

        case "memory":
            guard openModel(request) != nil else { return }
            let snapshot = Memory.snapshot()
            reply(
                id,
                ["active": snapshot.activeMemory, "cache": snapshot.cacheMemory, "peak": snapshot.peakMemory])

        case "trim":
            guard let model = openModel(request) else { return }
            guard request.bytes != nil else { return fail(id, "missing \"bytes\"") }
            let before = Memory.activeMemory + Memory.cacheMemory
            model.container?.cacheCoordinator?.releaseVolatile()  // kept prompt state (memory tier)
            Memory.clearCache()  // MLX buffer cache, process-wide
            let after = Memory.activeMemory + Memory.cacheMemory
            reply(id, ["freed": max(0, before - after)])

        default:
            fail(id, "unknown op \"\(request.op)\"")
        }
    }

    private func describe(_ error: Error) -> String {
        switch error {
        case DecodingError.keyNotFound(let key, _): return "missing \"\(key.stringValue)\""
        case DecodingError.typeMismatch(_, let context), DecodingError.valueNotFound(_, let context),
            DecodingError.dataCorrupted(let context):
            let path = context.codingPath.map(\.stringValue).joined(separator: ".")
            return path.isEmpty ? context.debugDescription : "\"\(path)\": \(context.debugDescription)"
        default: return "\(error)"
        }
    }

    /// The open, not-closing model a request names; replies with an error otherwise.
    private func openModel(_ request: Request) -> Model? {
        guard let handle = request.model else {
            fail(request.id, "missing \"model\"")
            return nil
        }
        guard let model = models[handle], !model.closing else {
            fail(request.id, "unknown model \(handle)")
            return nil
        }
        return model
    }

    private func firstInvalidToken(_ tokens: [Int], _ model: Model) -> Int? {
        let limit = model.facts.vocabularySize ?? Int(Int32.max)
        return tokens.first { $0 < 0 || $0 >= limit }
    }

    private func classify(_ directory: String) async -> ModelDirectoryInspection {
        let inspection = inspectModelDirectory(directory)
        guard case .model(let facts) = inspection else { return inspection }
        if knownModelTypes == nil {
            knownModelTypes = await LLMTypeRegistry.shared.registeredModelTypes
        }
        guard knownModelTypes?.contains(facts.modelType) == true else {
            return .notModel("model type \"\(facts.modelType)\" is not known to this engine")
        }
        return inspection
    }

    // MARK: Open / close

    private func open(id: UInt64, directory: String) async {
        let facts: ModelDirectoryFacts
        switch await classify(directory) {
        case .model(let found): facts = found
        case .notModel(let why): return fail(id, "not a loadable model directory: \(why)")
        }
        do {
            let container = try await loadModelContainer(from: facts.url, using: makeTokenizerLoader())
            // Prefix cache: memory tier only. The disk tier stays off, so nothing is written.
            await container.enableCachingAsync(
                config: CacheCoordinatorConfig(usePagedCache: true, enableDiskCache: false))
            let engine = await container.makeBatchEngine(maxBatchSize: maxBatchSize)
            let tokenizer = await container.tokenizer
            let handle = nextHandle
            nextHandle += 1
            let model = Model(
                handle: handle, facts: facts, container: container, engine: engine, tokenizer: tokenizer)
            if let tokenizerDirectory = try? await container.tokenizerDirectory {
                model.tokenizerShims = JangLoader.temporaryShimDirectories(behind: tokenizerDirectory)
            }
            models[handle] = model
            log("opened model \(handle): \(facts.url.path)")
            reply(id, ["model": handle])
        } catch {
            fail(id, "open failed: \(error)")
        }
    }

    /// Starts closing a model once; later callers share the same task.
    private func closeTask(for model: Model) -> Task<Void, Never> {
        if let existing = model.closeTask { return existing }
        model.closing = true
        let task = Task { await self.close(model) }
        model.closeTask = task
        return task
    }

    private func close(_ model: Model) async {
        // Cancel what is running; every request still sends its terminal message.
        let running = generations.values.filter { $0.model == model.handle }
        for generation in running {
            generation.cancelRequested = true
            if let batchID = generation.batchID { await model.engine?.cancel(batchID) }
        }
        for task in Array(model.tasks.values) { await task.value }

        await model.engine?.shutdown()
        model.container?.disableCaching()
        model.engine = nil
        model.tokenizer = nil
        model.container = nil
        for shim in model.tokenizerShims { try? FileManager.default.removeItem(at: shim) }
        models[model.handle] = nil
        Memory.clearCache()
        log("closed model \(model.handle)")
    }

    /// End of input: finish pending loads, then close every model.
    func shutdown() async {
        await loadChain?.value
        let closing = models.values.map { closeTask(for: $0) }
        for task in closing { await task.value }
    }

    // MARK: Generate

    private func startGeneration(_ request: Request) {
        let id = request.id
        guard let model = openModel(request) else { return }
        guard generations[id] == nil else {
            log("ignoring generate \(id): a request with this id is already in flight")
            return
        }
        guard let prompt = request.prompt else { return fail(id, "missing \"prompt\"") }
        guard !prompt.isEmpty else { return fail(id, "the prompt is empty") }
        if let bad = firstInvalidToken(prompt, model) {
            return fail(id, "prompt token \(bad) is outside the model's vocabulary")
        }
        if let temperature = request.temperature, !(temperature.isFinite && temperature >= 0) {
            return fail(id, "temperature must be a finite number >= 0")
        }
        if let topP = request.topP, !(topP.isFinite && topP >= 0 && topP <= 1) {
            return fail(id, "topP must be between 0 and 1")
        }

        // null maxTokens: until the model stops or its context is full.
        var maxTokens = request.maxTokens
        if maxTokens == nil, let contextLength = model.facts.contextLength {
            maxTokens = contextLength - prompt.count
        }
        if let maxTokens, maxTokens <= 0 {
            // Nothing is evaluated; the whole prompt is reported as uncached.
            out.send([
                "id": NSNumber(value: id), "event": "finished", "reason": "length",
                "promptTokens": prompt.count, "cachedPromptTokens": 0, "generatedTokens": 0,
                "promptSeconds": 0, "generationSeconds": 0, "stopToken": NSNull(),
            ])
            return
        }

        let generation = Generation(id: id, model: model.handle)
        generations[id] = generation
        model.tasks[id] = Task {
            await self.run(
                generation, on: model, prompt: prompt, maxTokens: maxTokens,
                temperature: request.temperature, topP: request.topP)
        }
    }

    private func run(
        _ generation: Generation, on model: Model, prompt: [Int], maxTokens: Int?,
        temperature: Double?, topP: Double?
    ) async {
        let id = NSNumber(value: generation.id)
        defer {
            generations[generation.id] = nil
            model.tasks[generation.id] = nil
        }
        guard let container = model.container, let engine = model.engine else {
            return fail(generation.id, "the model was closed")
        }

        // The model's own generation settings, then what the request overrides.
        var parameters = await container.defaultGenerateParameters()
        if let temperature { parameters.temperature = Float(temperature) }
        if let topP { parameters.topP = Float(topP) }
        parameters.maxTokens = maxTokens

        let (batchID, stream) = await engine.submit(
            input: LMInput(tokens: MLXArray(prompt), tokenIds: prompt), parameters: parameters)
        generation.batchID = batchID
        if generation.cancelRequested { await engine.cancel(batchID) }

        var delivered = 0
        var restored = 0
        var completion: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let token):
                guard completion == nil else { continue }
                delivered += 1
                out.send(["id": id, "event": "tokens", "tokens": [token]])
            case .prefillProgress(let progress):
                if progress.stage == .cacheRestore {
                    restored = max(restored, progress.completedUnitCount)
                }
            case .info(let info):
                completion = info
            }
        }

        guard let completion else {
            return fail(generation.id, "generation ended without a completion report")
        }
        if let failure = completion.generationFailure {
            return fail(generation.id, "generation failed: \(failure.cause)")
        }
        let reason: String
        switch completion.stopReason {
        case .stop: reason = "stop"
        case .length: reason = "length"
        case .cancelled:
            if generation.cancelRequested || model.closing {
                out.send(["id": id, "event": "cancelled"])
            } else {
                fail(generation.id, "the engine ended the generation early")
            }
            return
        }
        if completion.generationTokenCount != delivered {
            log(
                "generate \(generation.id): engine counted \(completion.generationTokenCount) tokens, delivered \(delivered)"
            )
        }
        let cached = min(restored, prompt.count)
        func seconds(_ value: TimeInterval) -> Double { value.isFinite ? max(0, value) : 0 }
        out.send([
            "id": id, "event": "finished", "reason": reason,
            "promptTokens": prompt.count - cached, "cachedPromptTokens": cached,
            "generatedTokens": delivered,
            "promptSeconds": seconds(completion.promptTime),
            "generationSeconds": seconds(completion.generateTime),
            "stopToken": completion.stopTokenID.map { $0 as Any } ?? NSNull(),
        ])
    }
}
