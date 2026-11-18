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
        .testTarget(
            name: "EngineRouterTests",
            dependencies: ["EngineRouter", "EngineRouterLocalCode"],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
    ]
)
