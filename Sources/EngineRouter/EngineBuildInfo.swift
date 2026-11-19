import Foundation
#if os(macOS)
import Darwin
#endif

/// Source identity for the build of an inference engine.
public struct EngineBuildInfo: Codable, Equatable, Sendable {
    /// The full source revision used to build the engine.
    public let revision: String
    /// A source tag that points to exactly `revision`, if known.
    public let tag: String?
    /// Whether the build included local source changes, if known.
    public let dirty: Bool?

    public init(revision: String, tag: String? = nil, dirty: Bool? = nil) {
        self.revision = revision
        self.tag = tag
        self.dirty = dirty
    }
}

#if os(macOS)
extension EngineBuildInfo {
    /// The optional native export returns a static NUL-terminated UTF-8 JSON string.
    package static func loadNative(from handle: UnsafeMutableRawPointer, symbol: String) -> Self? {
        typealias BuildInfo = @convention(c) () -> UnsafePointer<CChar>?
        guard let address = dlsym(handle, symbol) else { return nil }
        return decodeNative(unsafeBitCast(address, to: BuildInfo.self)())
    }

    static func decodeNative(_ json: UnsafePointer<CChar>?) -> Self? {
        guard let json, let string = String(validatingCString: json),
            let info = try? JSONDecoder().decode(Self.self, from: Data(string.utf8))
        else { return nil }

        let revisionLength = info.revision.utf8.count
        guard revisionLength == 40 || revisionLength == 64,
            info.revision.allSatisfy({ $0.isASCII && $0.isHexDigit })
        else { return nil }
        return info
    }
}
#endif
