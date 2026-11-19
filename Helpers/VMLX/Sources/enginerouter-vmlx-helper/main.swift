import Darwin
import EngineRouterVMLXService

// Standard output carries protocol frames and nothing else. Keep a private
// descriptor for the frames and point descriptor 1 at standard error, so a
// stray print() anywhere in the process cannot corrupt the stream.
let protocolFD = dup(STDOUT_FILENO)
guard protocolFD >= 0 else {
    fputs("enginerouter-vmlx-helper: could not duplicate standard output: \(String(cString: strerror(errno)))\n", stderr)
    exit(1)
}
dup2(STDERR_FILENO, STDOUT_FILENO)
// A client that goes away must surface as a write error, not a SIGPIPE death.
for fd in [STDOUT_FILENO, STDERR_FILENO] { _ = fcntl(fd, F_SETNOSIGPIPE, 1) }

exit(await VMLXHelper.serve(input: STDIN_FILENO, output: protocolFD))
