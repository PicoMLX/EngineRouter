// swift-tools-version: 6.1
import PackageDescription

// The helper process behind EngineRouterVMLX. It is a package of its own
// because vmlx-swift ships modules with the same names as the MLX libraries
// the EngineRouter package links, so the two cannot be in one build.
//
// Build it with xcodebuild, which also builds MLX's Metal library:
//
//     xcodebuild build -scheme enginerouter-vmlx-helper -configuration Release \
//       -destination 'platform=macOS,arch=arm64' -skipMacroValidation -skipPackagePluginValidation
//
// Keep `mlx-swift_Cmlx.bundle` from the build products beside the executable.
//
// An app in the App Store sandbox cannot start the executable. It adds an XPC
// Service target instead, which links the `EngineRouterVMLXService` library
// and whose main.swift calls `VMLXHelper.runXPCService()`; see EngineRouter's
// README.

// The PicoMLX fork of the engine the Osaurus app uses, on the branch that
// holds our changes until they land upstream.
let vmlxSwift: Package.Dependency = .package(url: "https://github.com/PicoMLX/vmlx-swift", branch: "engine-router")
let vmlxSwiftPackage = "vmlx-swift"

let package = Package(
    name: "enginerouter-vmlx-helper",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "enginerouter-vmlx-helper", targets: ["enginerouter-vmlx-helper"]),
        .library(name: "EngineRouterVMLXService", targets: ["EngineRouterVMLXService"]),
    ],
    dependencies: [vmlxSwift],
    targets: [
        .executableTarget(
            name: "enginerouter-vmlx-helper",
            dependencies: ["EngineRouterVMLXService"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "EngineRouterVMLXService",
            dependencies: [
                .product(name: "MLX", package: vmlxSwiftPackage),
                .product(name: "MLXLMCommon", package: vmlxSwiftPackage),
                .product(name: "MLXLLM", package: vmlxSwiftPackage),
                .product(name: "MLXHuggingFace", package: vmlxSwiftPackage),
                .product(name: "VMLXTokenizers", package: vmlxSwiftPackage),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
