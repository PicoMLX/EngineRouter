import Darwin
import Foundation

/// Largest frame accepted or produced, in bytes of JSON.
let maxFrameBytes = 64 * 1024 * 1024

/// Diagnostics go to standard error only; standard output carries frames.
func log(_ message: String) {
    let line = Array(("enginerouter-vmlx-helper: " + message + "\n").utf8)
    _ = line.withUnsafeBytes { write(STDERR_FILENO, $0.baseAddress, $0.count) }
}

/// Writes length-prefixed JSON frames. Safe to call from any task: one lock
/// covers a whole frame, so frames never interleave.
final class FrameWriter: @unchecked Sendable {
    private let fd: Int32
    private let lock = NSLock()
    private var broken = false

    init(fd: Int32) { self.fd = fd }

    func send(_ object: [String: Any]) {
        var body: Data
        do {
            body = try JSONSerialization.data(withJSONObject: object)
        } catch {
            log("could not encode a response: \(error)")
            return
        }
        if body.count > maxFrameBytes, let id = object["id"] {
            log("response of \(body.count) bytes exceeds the frame limit; sending an error instead")
            body =
                (try? JSONSerialization.data(withJSONObject: [
                    "id": id, "error": "response exceeds the 64 MiB frame limit",
                ])) ?? Data()
        }
        var frame = Data(capacity: body.count + 4)
        var length = UInt32(body.count).bigEndian
        withUnsafeBytes(of: &length) { frame.append(contentsOf: $0) }
        frame.append(body)

        lock.lock()
        defer { lock.unlock() }
        guard !broken else { return }
        frame.withUnsafeBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                let written = write(fd, buffer.baseAddress! + offset, buffer.count - offset)
                if written < 0 {
                    if errno == EINTR { continue }
                    log("write to the client failed: \(String(cString: strerror(errno)))")
                    broken = true
                    return
                }
                offset += written
            }
        }
    }
}

enum Incoming {
    case frame(Data)
    case endOfInput
    case truncated
    case tooLarge(Int)
}

/// Reads frames from `fd` on a dedicated thread (blocking reads) and hands
/// them over in arrival order.
func incomingFrames(fd: Int32) -> AsyncStream<Incoming> {
    AsyncStream { continuation in
        let thread = Thread {
            /// Returns nil at end of input; `complete` is false when it ended mid-read.
            func read(_ count: Int) -> (data: Data, complete: Bool) {
                var data = Data(count: count)
                var filled = 0
                while filled < count {
                    let got = data.withUnsafeMutableBytes {
                        Darwin.read(fd, $0.baseAddress! + filled, count - filled)
                    }
                    if got < 0 {
                        if errno == EINTR { continue }
                        log("read failed: \(String(cString: strerror(errno)))")
                        return (data.prefix(filled), false)
                    }
                    if got == 0 { return (data.prefix(filled), false) }
                    filled += got
                }
                return (data, true)
            }
            while true {
                let header = read(4)
                guard header.complete else {
                    continuation.yield(header.data.isEmpty ? .endOfInput : .truncated)
                    break
                }
                let length = header.data.reduce(0) { ($0 << 8) | Int($1) }
                guard length <= maxFrameBytes else {
                    continuation.yield(.tooLarge(length))
                    break
                }
                let body = read(length)
                guard body.complete else {
                    continuation.yield(.truncated)
                    break
                }
                continuation.yield(.frame(body.data))
            }
            continuation.finish()
        }
        thread.name = "enginerouter-vmlx-helper.stdin"
        thread.start()
    }
}
