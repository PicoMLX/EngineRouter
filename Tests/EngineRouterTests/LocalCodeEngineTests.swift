#if os(macOS)
import EngineRouter
import Foundation
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
        #expect(files.engram.lastPathComponent == "model_i2x.engram")
        #expect(files.drafter == nil)
    }

    @Test
    func aSecondCheckpointNamedDSparkIsTheDrafter() throws {
        let directory = try directory(["model_i2x.ckpt", "model_i2x.engram", "model_dspark_i4x.ckpt"])
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(ModelFiles(directory: directory)?.drafter?.lastPathComponent == "model_dspark_i4x.ckpt")
    }

    @Test(arguments: [
        ["model.ckpt"],  // no engram file
        ["model.engram"],  // no checkpoint
        ["a.engram", "b.ckpt"],  // names do not match
        ["a.engram", "a.ckpt", "b.engram"],  // two engram files
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

private enum LiveLocalCode {
    static var library: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_LOCAL_CODE_LIBRARY"] }
    static var modelDirectory: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_LOCAL_CODE_MODEL_DIR"] }
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

        // A greedy request runs to the model's stop token.
        let (tokens, completion) = try await run(
            loaded, EngineRequest(promptTokens: prompt, maxTokens: 64, temperature: 0))
        let answer = try await loaded.detokenize(tokens)
        print("Local Code answered: \(answer)")
        let finished = try #require(completion)
        #expect(answer.contains("12") || answer.localizedCaseInsensitiveContains("twelve"))
        #expect(finished.reason == .stop)
        #expect(finished.generatedTokenCount == tokens.count)
        #expect(finished.promptTokenCount == prompt.count)
        #expect(await loaded.memory().active == status.loadBytes)

        // The token limit ends a request early.
        let limited = try await run(loaded, EngineRequest(promptTokens: prompt, maxTokens: 2, temperature: 0))
        #expect(limited.completion?.reason == .length)
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
