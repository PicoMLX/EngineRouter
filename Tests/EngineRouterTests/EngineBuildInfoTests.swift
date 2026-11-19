#if os(macOS)
import Darwin
@testable import EngineRouter
import Testing

struct EngineBuildInfoTests {
    private let revision = "0123456789abcdef0123456789abcdef01234567"

    @Test
    func readsNativeBuildInfo() {
        let json = #"{"revision":"\#(revision)","tag":"v1.2.3","dirty":false}"#
        #expect(json.withCString(EngineBuildInfo.decodeNative) ==
            EngineBuildInfo(revision: revision, tag: "v1.2.3", dirty: false))
    }

    @Test
    func acceptsAnUntaggedNativeBuild() {
        let json = #"{"revision":"\#(revision)","dirty":true}"#
        #expect(json.withCString(EngineBuildInfo.decodeNative) == EngineBuildInfo(revision: revision, dirty: true))
    }

    @Test(arguments: ["{}", "invalid", #"{"revision":"bad"}"#, #"{"revision":"no source revision available for a release"}"#])
    func ignoresInvalidNativeBuildInfo(json: String) {
        #expect(json.withCString(EngineBuildInfo.decodeNative) == nil)
    }

    @Test
    func allowsLibrariesWithoutBuildMetadata() throws {
        let handle = try #require(dlopen(nil, RTLD_NOW))
        defer { dlclose(handle) }
        #expect(EngineBuildInfo.loadNative(from: handle, symbol: "pico_build_info_absent_fixture") == nil)
        #expect(EngineBuildInfo.decodeNative(nil) == nil)
    }
}
#endif
