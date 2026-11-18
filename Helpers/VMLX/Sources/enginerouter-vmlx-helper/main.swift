import Darwin
import Foundation

// Standard output carries protocol frames and nothing else. Keep a private
// descriptor for the frames and point descriptor 1 at standard error, so a
// stray print() anywhere in the process cannot corrupt the stream.
let protocolFD = dup(STDOUT_FILENO)
guard protocolFD >= 0 else {
    log("could not duplicate standard output: \(String(cString: strerror(errno)))")
    exit(1)
}
dup2(STDERR_FILENO, STDOUT_FILENO)
// A client that goes away must surface as a write error, not a SIGPIPE death.
for fd in [protocolFD, STDOUT_FILENO, STDERR_FILENO] { _ = fcntl(fd, F_SETNOSIGPIPE, 1) }

let server = Server(out: FrameWriter(fd: protocolFD))
var status: Int32 = 0

loop: for await incoming in incomingFrames(fd: STDIN_FILENO) {
    switch incoming {
    case .frame(let data):
        await server.handle(data)
    case .endOfInput:
        break loop
    case .truncated:
        log("standard input ended in the middle of a frame; shutting down")
        break loop
    case .tooLarge(let length):
        log("error: frame of \(length) bytes exceeds the \(maxFrameBytes)-byte limit; exiting")
        status = 1
        break loop
    }
}

await server.shutdown()
exit(status)
