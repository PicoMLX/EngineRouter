import Darwin
import Foundation
import XPC

/// The two ways the helper runs: as an executable over its standard input and
/// output, or as an XPC service that an app embeds in its bundle. The App
/// Store sandbox allows only the service. Both speak the same messages.
public enum VMLXHelper {
    /// Serves one engine: reads requests from `input` and writes answers to
    /// `output` until `input` ends, then closes every model it opened.
    ///
    /// - Returns: The status the executable exits with.
    public static func serve(input: Int32, output: Int32) async -> Int32 {
        // A client that goes away must surface as a write error, not a SIGPIPE death.
        _ = fcntl(output, F_SETNOSIGPIPE, 1)
        let server = Server(out: FrameWriter(fd: output))
        var status: Int32 = 0
        loop: for await incoming in incomingFrames(fd: input) {
            switch incoming {
            case .frame(let data):
                await server.handle(data)
            case .endOfInput:
                break loop
            case .truncated:
                log("the input ended in the middle of a frame; shutting down")
                break loop
            case .tooLarge(let length):
                log("error: frame of \(length) bytes exceeds the \(maxFrameBytes)-byte limit; stopping")
                status = 1
                break loop
            }
        }
        await server.shutdown()
        return status
    }

    /// Runs the XPC service; never returns. The `main.swift` of an app's XPC
    /// Service target calls this and nothing else.
    public static func runXPCService() -> Never {
        let host = SessionHost()
        let listener = NSXPCListener.service()
        listener.delegate = host
        // A service listener runs the process from here on.
        listener.resume()
        fatalError("NSXPCListener.service() stopped listening")
    }
}

/// The service's side of a session. EngineRouterVMLX declares the same
/// protocol: the two packages cannot share a module, and XPC matches the
/// method, not the protocol's name. Change both together.
@objc(EngineRouterVMLXHelperXPC)
protocol VMLXHelperXPC {
    /// Serves one engine: reads requests from `input` and writes answers to
    /// `output` until `input` ends. Replies once the session has started.
    func serve(input: FileHandle, output: FileHandle, reply: @escaping () -> Void)
}

/// Accepts the app's connections and serves each session on its own. The
/// sessions share the process's MLX, and their model loads take turns.
final class SessionHost: NSObject, NSXPCListenerDelegate, VMLXHelperXPC, @unchecked Sendable {
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        connection.exportedInterface = NSXPCInterface(with: VMLXHelperXPC.self)
        connection.exportedObject = self
        connection.resume()
        return true
    }

    func serve(input: FileHandle, output: FileHandle, reply: @escaping () -> Void) {
        // Once the reply is sent no message is in flight, and the system may
        // end an idle service; the session's transaction keeps it running.
        xpc_transaction_begin()
        reply()
        Task.detached {
            _ = await VMLXHelper.serve(input: input.fileDescriptor, output: output.fileDescriptor)
            try? input.close()
            try? output.close()
            xpc_transaction_end()
        }
    }
}
