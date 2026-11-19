#if os(macOS)
import Darwin
import Dispatch
import EngineRouter
import Foundation

/// The Local Code dynamic library and its C entry points.
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

    /// Receives one Chat Completions chunk as JSON text, or nil to ask whether
    /// to go on. Returns 0 to cancel.
    typealias ChatCallback = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Int32
    typealias Chat = @convention(c) (
        UnsafeMutableRawPointer?, UnsafePointer<CChar>?, ChatCallback?, UnsafeMutableRawPointer?
    ) -> Int32
    typealias CountChatTokens = @convention(c) (
        UnsafeMutableRawPointer?, UnsafePointer<CChar>?, UnsafeMutablePointer<CChar>?, Int32
    ) -> Int32
    typealias Trim = @convention(c) (UnsafeMutableRawPointer?) -> Int64

    /// Local Code keeps one process-wide runtime and its calls block, so they
    /// run one at a time, off Swift's cooperative executor.
    static let queue = DispatchQueue(label: "enginerouter.local-code", qos: .userInitiated)

    /// Opens DeepSeek 4.1 only; it needs the Engram file.
    let open: Open?
    /// Opens every family the library serves; a library built before GLM 5.3 Flash lacks it.
    let openAnyFamily: Open?
    let close: Close
    let residentBytes: ResidentBytes
    let tokenize: Tokenize
    let detokenize: Detokenize
    let generateTokens: GenerateTokens
    /// The chat-level entry points, which a library built before them lacks.
    let chat: Chat?
    let countChatTokens: CountChatTokens?
    let trim: Trim?
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
        func optionalSymbol<T>(_ name: String) -> T? {
            dlsym(handle, name).map { unsafeBitCast($0, to: T.self) }
        }
        open = try symbol("pico_local_code_open_v2") as Open
        openAnyFamily = optionalSymbol("pico_local_code_open_v3")
        close = try symbol("pico_local_code_close_v2")
        residentBytes = try symbol("pico_local_code_resident_bytes_v2")
        tokenize = try symbol("pico_local_code_tokenize_v3")
        detokenize = try symbol("pico_local_code_detokenize_v2")
        generateTokens = try symbol("pico_local_code_generate_tokens_v3")
        chat = optionalSymbol("pico_local_code_chat_v1")
        countChatTokens = optionalSymbol("pico_local_code_count_chat_tokens_v1")
        trim = optionalSymbol("pico_local_code_trim_v1")
        buildInfo = .loadNative(from: handle, symbol: "pico_local_code_build_info_v1")
    }

    /// The entry points themselves, for tests that have no library file.
    init(
        open: Open, openAnyFamily: Open? = nil, close: Close, residentBytes: ResidentBytes, tokenize: Tokenize,
        detokenize: Detokenize,
        generateTokens: GenerateTokens, chat: Chat? = nil, countChatTokens: CountChatTokens? = nil, trim: Trim? = nil
    ) {
        self.open = open
        self.openAnyFamily = openAnyFamily
        self.close = close
        self.residentBytes = residentBytes
        self.tokenize = tokenize
        self.detokenize = detokenize
        self.generateTokens = generateTokens
        self.chat = chat
        self.countChatTokens = countChatTokens
        self.trim = trim
        self.buildInfo = nil
    }

    /// The error for a chat call on a library built before the chat entry points.
    static let noChat = LocalCodeError.library(
        "The Local Code library has no chat entry points. Rebuild it from the fork.")

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
    /// A token-level call on a model served through `chat` only.
    case chatOnly
    case generationFailed
    case closed

    public var description: String {
        switch self {
        case .library(let message), .model(let message): message
        case .invalidText: "Local Code could not read the text as UTF-8."
        case .chatOnly: "This Local Code model is served through chat only: its tokens are not DeepSeek 4.1's."
        case .generationFailed: "Local Code failed to generate."
        case .closed: "The Local Code model is closed."
        }
    }
}
#endif
