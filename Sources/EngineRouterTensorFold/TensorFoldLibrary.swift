#if os(macOS)
import Darwin
import Dispatch
import Foundation

/// The TensorFold dynamic library and its C entry points.
///
/// The function pointers are immutable, so the type is safe to share.
final class TensorFoldLibrary: @unchecked Sendable {
    /// Called on the engine's thread. `status` is -1 for a batch of tokens.
    /// Any other value ends the request and comes with `usage`: 0 stop,
    /// 1 cancelled, 2 failed, 3 length.
    typealias EventCallback = @convention(c) (
        UnsafeMutableRawPointer?, UnsafePointer<Int32>?, Int32, Int32, UnsafePointer<Int64>?
    ) -> Void

    typealias CanServe = @convention(c) (UnsafePointer<CChar>?) -> Int32
    typealias LoadBytes = @convention(c) (UnsafePointer<CChar>?) -> Int64
    typealias Open = @convention(c) (UnsafePointer<CChar>?, UnsafeMutablePointer<CChar>?, Int32) -> UnsafeMutableRawPointer?
    typealias Close = @convention(c) (UnsafeMutableRawPointer?) -> Void
    /// The text is UTF-8 bytes with a length, so it can hold NUL characters.
    typealias Tokenize = @convention(c) (
        UnsafeMutableRawPointer?, UnsafePointer<UInt8>?, Int32, UnsafeMutablePointer<Int32>?, Int32
    ) -> Int32
    typealias Detokenize = @convention(c) (
        UnsafeMutableRawPointer?, UnsafePointer<Int32>?, Int32, UnsafeMutablePointer<CChar>?, Int32
    ) -> Int32
    typealias Submit = @convention(c) (
        UnsafeMutableRawPointer?, UInt64, UnsafePointer<Int32>?, Int32, Int32, Float, Float, EventCallback?,
        UnsafeMutableRawPointer?
    ) -> Int32
    typealias Cancel = @convention(c) (UnsafeMutableRawPointer?, UInt64) -> Void
    typealias Memory = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<Int64>?) -> Int32

    /// The version of the library's C interface this adapter is written against.
    static let interfaceVersion: Int32 = 3

    /// Opening and closing a model block, so they run off Swift's cooperative executor.
    static let queue = DispatchQueue(label: "enginerouter.tensorfold", qos: .userInitiated)

    let canServe: CanServe
    let loadBytes: LoadBytes
    let open: Open
    let close: Close
    let tokenize: Tokenize
    let detokenize: Detokenize
    let submit: Submit
    let cancel: Cancel
    let memory: Memory

    init(url: URL) throws {
        guard url.isFileURL else { throw TensorFoldError.library("The TensorFold library must be a local file.") }
        // The engine's thread and Metal state can outlive a handle, so the
        // library is never unloaded.
        guard let handle = url.withUnsafeFileSystemRepresentation({ path in
            path.flatMap { dlopen($0, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE) }
        }) else {
            throw TensorFoldError.library(dlerror().map { String(cString: $0) } ?? "The library did not load.")
        }
        func symbol<T>(_ name: String) throws -> T {
            guard let address = dlsym(handle, name) else {
                throw TensorFoldError.library("The TensorFold library has no \(name). Rebuild it from the fork.")
            }
            return unsafeBitCast(address, to: T.self)
        }
        // An entry point that changes its arguments keeps its name, so the version is what tells them apart.
        let version: @convention(c) () -> Int32 = try symbol("pico_tf_abi_version")
        guard version() == Self.interfaceVersion else {
            throw TensorFoldError.library(
                "The TensorFold library has interface version \(version()); this adapter needs "
                    + "\(Self.interfaceVersion). Rebuild it from the fork.")
        }
        canServe = try symbol("pico_tf_can_serve")
        loadBytes = try symbol("pico_tf_load_bytes")
        open = try symbol("pico_tf_open")
        close = try symbol("pico_tf_close")
        tokenize = try symbol("pico_tf_tokenize")
        detokenize = try symbol("pico_tf_detokenize")
        submit = try symbol("pico_tf_submit")
        cancel = try symbol("pico_tf_cancel")
        memory = try symbol("pico_tf_memory")
    }

    /// The entry points themselves, for tests that have no library file.
    init(
        canServe: CanServe, loadBytes: LoadBytes, open: Open, close: Close, tokenize: Tokenize,
        detokenize: Detokenize, submit: Submit, cancel: Cancel, memory: Memory
    ) {
        self.canServe = canServe
        self.loadBytes = loadBytes
        self.open = open
        self.close = close
        self.tokenize = tokenize
        self.detokenize = detokenize
        self.submit = submit
        self.cancel = cancel
        self.memory = memory
    }

    /// Runs a blocking library call on ``queue``.
    func onQueue<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            Self.queue.async { continuation.resume(with: Result { try body() }) }
        }
    }
}

public enum TensorFoldError: Error, Equatable, CustomStringConvertible {
    /// The dynamic library is missing, unloadable or too old.
    case library(String)
    /// The model's files are missing or are not a model TensorFold runs.
    case model(String)
    case invalidText
    case generationFailed
    case closed

    public var description: String {
        switch self {
        case .library(let message), .model(let message): message
        case .invalidText: "TensorFold could not read the text as UTF-8."
        case .generationFailed: "TensorFold failed to generate."
        case .closed: "The TensorFold model is closed."
        }
    }
}
#endif
