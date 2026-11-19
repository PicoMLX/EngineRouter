#if os(macOS)
import Darwin
import Dispatch
import EngineRouter
import Foundation

/// The Local Code dynamic library and its token-level C entry points.
///
/// The function pointers and the handle are immutable, so the type is safe to
/// share. Every call that touches a model goes through ``queue``.
final class LocalCodeLibrary: @unchecked Sendable {
    typealias TokenCallback = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Int32>?, Int32) -> Int32

    typealias Open = @convention(c) (
        UnsafePointer<CChar>?, UnsafePointer<CChar>?, UnsafePointer<CChar>?, UnsafeMutablePointer<CChar>?, Int32
    ) -> UnsafeMutableRawPointer?
    typealias Close = @convention(c) (UnsafeMutableRawPointer?) -> Void
    typealias ResidentBytes = @convention(c) (UnsafeMutableRawPointer?) -> Int64
    /// The text is UTF-8 bytes with a length, so it can hold NUL characters.
    typealias Tokenize = @convention(c) (
        UnsafePointer<UInt8>?, Int32, UnsafeMutablePointer<Int32>?, Int32
    ) -> Int32
    typealias Detokenize = @convention(c) (
        UnsafePointer<Int32>?, Int32, UnsafeMutablePointer<CChar>?, Int32
    ) -> Int32
    typealias GenerateTokens = @convention(c) (
        UnsafeMutableRawPointer?, UnsafePointer<Int32>?, Int32, Int32, Int32, TokenCallback?,
        UnsafeMutableRawPointer?, UnsafeMutablePointer<Int64>?
    ) -> Int32

    /// Local Code keeps one process-wide runtime and its calls block, so they
    /// run one at a time, off Swift's cooperative executor.
    static let queue = DispatchQueue(label: "enginerouter.local-code", qos: .userInitiated)

    let open: Open
    let close: Close
    let residentBytes: ResidentBytes
    let tokenize: Tokenize
    let detokenize: Detokenize
    let generateTokens: GenerateTokens
    let buildInfo: EngineBuildInfo?

    init(url: URL) throws {
        guard url.isFileURL else { throw LocalCodeError.library("The Local Code library must be a local file.") }
        // Swift and Metal runtime state inside the library can outlive a handle,
        // so the library is never unloaded.
        guard let handle = url.withUnsafeFileSystemRepresentation({ path in
            path.flatMap { dlopen($0, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE) }
        }) else {
            throw LocalCodeError.library(dlerror().map { String(cString: $0) } ?? "The library did not load.")
        }
        func symbol<T>(_ name: String) throws -> T {
            guard let address = dlsym(handle, name) else {
                throw LocalCodeError.library("The Local Code library has no \(name). Rebuild it from the fork.")
            }
            return unsafeBitCast(address, to: T.self)
        }
        open = try symbol("pico_local_code_open_v2")
        close = try symbol("pico_local_code_close_v2")
        residentBytes = try symbol("pico_local_code_resident_bytes_v2")
        tokenize = try symbol("pico_local_code_tokenize_v3")
        detokenize = try symbol("pico_local_code_detokenize_v2")
        generateTokens = try symbol("pico_local_code_generate_tokens_v3")
        buildInfo = .loadNative(from: handle, symbol: "pico_local_code_build_info_v1")
    }

    /// The entry points themselves, for tests that have no library file.
    init(
        open: Open, close: Close, residentBytes: ResidentBytes, tokenize: Tokenize, detokenize: Detokenize,
        generateTokens: GenerateTokens
    ) {
        self.open = open
        self.close = close
        self.residentBytes = residentBytes
        self.tokenize = tokenize
        self.detokenize = detokenize
        self.generateTokens = generateTokens
        self.buildInfo = nil
    }

    /// Runs a blocking library call on ``queue``.
    func onQueue<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            Self.queue.async { continuation.resume(with: Result { try body() }) }
        }
    }
}

public enum LocalCodeError: Error, Equatable, CustomStringConvertible {
    /// The dynamic library is missing, unloadable or too old.
    case library(String)
    /// The model's files are missing or are not a Local Code checkpoint.
    case model(String)
    case invalidText
    case generationFailed
    case closed

    public var description: String {
        switch self {
        case .library(let message), .model(let message): message
        case .invalidText: "Local Code could not read the text as UTF-8."
        case .generationFailed: "Local Code failed to generate."
        case .closed: "The Local Code model is closed."
        }
    }
}
#endif
