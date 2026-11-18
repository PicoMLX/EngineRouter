#if os(macOS)
import Darwin
import EngineRouter
import EngineRouterMLX
import Foundation
import Synchronization
import Testing

private enum LiveCoexistence {
    static var library: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_MLXSERVE_LIBRARY"] }
    static var modelDirectory: String? { ProcessInfo.processInfo.environment["ENGINEROUTER_MLX_MODEL_DIR"] }
}

/// Whether a library that carries its own private copy of MLX can run in a
/// process where Swift MLX is running too. The mlx-serve adapter depends on
/// the answer.
///
/// It needs the probe library built from the mlx-serve fork and an MLX model,
/// and runs with `xcodebuild`, as every test that uses Swift MLX does:
///
///     TEST_RUNNER_ENGINEROUTER_MLXSERVE_LIBRARY=/path/to/libPicoMlxServe.dylib \
///     TEST_RUNNER_ENGINEROUTER_MLX_MODEL_DIR=/path/to/model \
///     xcodebuild test -scheme EngineRouter-Package -destination 'platform=macOS' \
///       -only-testing:EngineRouterTests/MLXCoexistenceTests
@Suite(.enabled(if: LiveCoexistence.library != nil && LiveCoexistence.modelDirectory != nil), .serialized)
struct MLXCoexistenceTests {
    private typealias Smoke = @convention(c) (Int32, Int32) -> Double
    private typealias Memory = @convention(c) (UnsafeMutablePointer<Int64>?) -> Int32
    private typealias Version = @convention(c) () -> UnsafePointer<CChar>?

    @Test(.timeLimit(.minutes(10)))
    func aSecondCopyOfMLXRunsBesideSwiftMLX() async throws {
        // Swift MLX first, on its own: the tokens every later run must repeat.
        let engine = MLXEngine()
        let model = EngineModel(
            id: "coexistence", directory: URL(fileURLWithPath: try #require(LiveCoexistence.modelDirectory)))
        let loaded = try await engine.open(model)
        let prompt = try await loaded.tokenize("The capital of France is")
        let baseline = try await tokens(loaded, prompt, limit: 48)
        #expect(baseline.count == 48)

        // The second copy, loaded into the same process.
        let handle = try #require(dlopen(try #require(LiveCoexistence.library), RTLD_NOW | RTLD_LOCAL))
        let smoke = unsafeBitCast(try #require(dlsym(handle, "pico_ms_smoke")), to: Smoke.self)
        let memory = unsafeBitCast(try #require(dlsym(handle, "pico_ms_memory")), to: Memory.self)
        let version = unsafeBitCast(try #require(dlsym(handle, "pico_ms_mlx_version")), to: Version.self)
        print("The library's MLX is \(version().map { String(cString: $0) } ?? "unknown")")

        // Alone after Swift MLX has run: n x n of 1/n, multiplied by itself, sums to n.
        #expect(abs(smoke(512, 8) - 512) < 1e-3)
        // And Swift MLX is undisturbed by it.
        #expect(try await tokens(loaded, prompt, limit: 48) == baseline)

        // Both at once: four threads in the library's MLX while Swift MLX generates.
        let wrong = Mutex(0)
        let calls = Mutex(0)
        let threads = (0..<4).map { _ in
            Thread {
                for _ in 0..<25 {
                    let ok = abs(smoke(1024, 6) - 1024) < 1e-2
                    calls.withLock { $0 += 1 }
                    if !ok { wrong.withLock { $0 += 1 } }
                }
            }
        }
        for thread in threads { thread.start() }
        var generations = 0
        while threads.contains(where: { !$0.isFinished }) || generations < 3 {
            #expect(try await tokens(loaded, prompt, limit: 48) == baseline)
            generations += 1
        }
        #expect(calls.withLock { $0 } == 100)
        #expect(wrong.withLock { $0 } == 0)
        print("Swift MLX generated \(generations) times while the library ran 100 computations")

        // Each copy keeps its own books.
        var bytes = [Int64](repeating: 0, count: 3)
        #expect(memory(&bytes) == 0)
        let swift = await loaded.memory()
        print("Library MLX: active \(bytes[0]), cache \(bytes[1]), peak \(bytes[2])")
        print("Swift MLX: weights \(swift.active), cache \(swift.cache), peak \(swift.peak)")
        #expect(bytes[2] > 0)
        #expect(Int(bytes[2]) != swift.peak)

        await loaded.close()
        // The library still works once Swift MLX's model is gone.
        #expect(abs(smoke(512, 8) - 512) < 1e-3)
    }

    private func tokens(_ loaded: any LoadedModel, _ prompt: [Int], limit: Int) async throws -> [Int] {
        var tokens: [Int] = []
        for try await event in loaded.generate(EngineRequest(promptTokens: prompt, maxTokens: limit, temperature: 0)) {
            if case .tokens(let ids) = event { tokens += ids }
        }
        return tokens
    }
}
#endif
