// swift-tools-version: 6.4
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "EngineRouter",
    // The same floor as PicoCore, the first app layer built on this package.
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "EngineRouter",
            targets: ["EngineRouter"]
        ),
        // macOS only. Loads a Local Code library you build yourself; see the
        // licence note in the README.
        .library(
            name: "EngineRouterLocalCode",
            targets: ["EngineRouterLocalCode"]
        ),
        // macOS only. Loads a TensorFold library you build from the fork.
        .library(
            name: "EngineRouterTensorFold",
            targets: ["EngineRouterTensorFold"]
        ),
        // macOS only. Loads an mlx-serve library you build from the fork.
        .library(
            name: "EngineRouterMLXServe",
            targets: ["EngineRouterMLXServe"]
        ),
        // MLX models, run with mlx-swift-lm.
        .library(
            name: "EngineRouterMLX",
            targets: ["EngineRouterMLX"]
        ),
    ],
    dependencies: [
        // The PicoMLX forks, on the branch PicoCore builds against. SwiftPM
        // wants one branch requirement for a package across the whole graph,
        // and mlx-swift-lm's manifest names mlx-swift's, so these two and
        // PicoCore's move together.
        .package(url: "https://github.com/PicoMLX/mlx-swift-lm", branch: "pico/integration"),
        .package(url: "https://github.com/PicoMLX/mlx-swift", branch: "pico/integration"),
        // The tokenizer loader macro in MLXHuggingFace expands to code that uses Tokenizers.
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.4.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "EngineRouter",
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .target(
            name: "EngineRouterLocalCode",
            dependencies: ["EngineRouter"],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .target(
            name: "EngineRouterTensorFold",
            dependencies: ["EngineRouter"],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .target(
            name: "EngineRouterMLXServe",
            dependencies: ["EngineRouter"],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .target(
            name: "EngineRouterMLX",
            dependencies: [
                "EngineRouter",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .testTarget(
            name: "EngineRouterTests",
            dependencies: [
                "EngineRouter", "EngineRouterLocalCode", "EngineRouterMLX", "EngineRouterMLXServe",
                "EngineRouterTensorFold",
            ],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
    ]
)
