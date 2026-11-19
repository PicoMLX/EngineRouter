# EngineRouter

EngineRouter hosts on-device inference engines inside one app process and
routes each model to the engine that can serve it.

- An **engine** keeps its own batching, scheduling and KV cache.
- The **router** decides which engine serves a model, loads and unloads
  models, and holds one memory budget across all engines.
- The **app** above it keeps chat templates, the tool loop, users and
  conversations. For Pico that app layer is
  [PicoCore](https://github.com/PicoMLX/picocore).

## Status

Early. The engine interface, the chat-level interface, the router, and the
MLX, Local Code, TensorFold, mlx-serve and vmlx adapters exist.

## Using it

```swift
import EngineRouter
import EngineRouterLocalCode

let router = Router(
    engines: [try LocalCodeEngine(libraryURL: libraryURL)],
    memoryBudget: 96 << 30)

let model = try await router.open(EngineModel(id: "deepseek-4.1-flash", directory: modelDirectory))
let prompt = try await model.tokenize("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")
var reply: [Int] = []
for try await event in model.generate(EngineRequest(promptTokens: prompt, maxTokens: 256)) {
    switch event {
    case .tokens(let ids): reply += ids
    case .finished(let completion):
        print(try await model.detokenize(reply))
        print("\(completion.generatedTokenCount) tokens")
    }
}
await router.close(model.id)
```

A completion whose reason is `.stop` names the token that stopped the reply in
`stopToken`, for chat formats that end a reply with different tokens for
different outcomes.

A character can span several tokens, and half of one decodes to nothing or to
U+FFFD, so decode tokens together, never one event at a time. To show text as
it arrives, decode the reply so far and emit what that adds to the text
already shown.

Cancelling the task that consumes a request's stream cancels the request.

### Chat-level models

An engine that already formats prompts and reads replies for its models
offers them as `ChatModel`s. The app sends the whole conversation and the
tools it offers; the engine lays them out in the model's own chat format,
splits the reply into reasoning, text and tool calls, and keeps the caches
that let a conversation's next turn skip the prompt it has already evaluated.
The app runs the tools and sends their results in the next request.

```swift
if let chat = model as? any ChatModel {
    let request = ChatRequest(
        messages: [ChatMessage(role: .user, content: "What is the weather in Paris?")],
        tools: [ChatTool(name: "get_weather", parameters: #"{"type":"object"}"#)])
    for try await event in chat.chat(request) {
        switch event {
        case .reasoning(let text), .text(let text): print(text, terminator: "")
        case .toolCall(let call): print("\n\(call.name)(\(call.arguments))")
        case .finished(let completion): print("\n\(completion.cachedPromptTokenCount) cached prompt tokens")
        }
    }
}
```

`countPromptTokens` counts a request's prompt in the same layout. Between the
adapter and its engine the request travels as an OpenAI Chat Completions body
and the reply as its streamed chunks. So far Local Code's models are chat
models; the others are token-level.

`Router.open` goes to the first engine whose `canServe` accepts the model and
charges the model's `loadBytes` against the budget. It throws when the model
does not fit or when an exclusive engine is involved; the router never closes
a model on its own, so the caller decides what to close.

## Engines

| Engine | How it is sourced | Licence | State |
|---|---|---|---|
| [mlx-swift-lm](https://github.com/PicoMLX/mlx-swift-lm) | Our fork, on the same branch requirement PicoCore uses | MIT | `EngineRouterMLX`, text only, one request at a time per call; see below |
| [Local Code](https://github.com/drawthingsai/local-code-community) | [Our fork](https://github.com/PicoMLX/local-code-community/tree/engine-router), built locally into a dynamic library | GPL-3.0 | `EngineRouterLocalCode`, macOS only, DeepSeek 4.1 checkpoints; see the licence note |
| [mlx-serve](https://github.com/ddalcu/mlx-serve) | [Our fork](https://github.com/PicoMLX/mlx-serve/tree/engine-router), built locally into a dynamic library with its own copy of MLX | MIT | `EngineRouterMLXServe`, macOS 26.2 or later, text only; see below |
| [Sushi](https://github.com/beamivalice/sushi) EXL3 | Through mlx-serve's vendored EXL3 module | Via mlx-serve | In the mlx-serve library, not yet run with an EXL3 model |
| [TensorFold](https://github.com/ashhart/TensorFold) | [Our fork](https://github.com/PicoMLX/TensorFold/tree/engine-router), built locally into a dynamic library | Apache-2.0 | `EngineRouterTensorFold`, macOS only; so far run only against a synthetic checkpoint, see below |
| [Osaurus](https://github.com/osaurus-ai/osaurus)'s engine, [vmlx-swift](https://github.com/osaurus-ai/vmlx-swift) | [Our fork](https://github.com/PicoMLX/vmlx-swift/tree/engine-router), built into a helper process | MIT | `EngineRouterVMLX`, macOS only, text only, out of process (helper executable or XPC service); see below |

Each adapter is its own library product. An app that depends only on
`EngineRouter` links no engine.

## Running MLX models

`EngineRouterMLX` runs a model directory in the layout `mlx-community` models
use: `config.json`, the tokenizer files and the weights as `.safetensors`
files.

```swift
import EngineRouterMLX

let router = Router(engines: [MLXEngine()], memoryBudget: 96 << 30)
```

- A model's `loadBytes` is the size of its weight files.
- Each request starts from an empty KV cache and requests are not batched
  together. PicoCore's own MLX path batches; this adapter does not yet.
- MLX keeps one buffer cache for the whole process, so every open MLX model
  reports the same `cache`, and `trim` drops all of it.
- A request that sets no `temperature` or `topP` uses the model's own
  generation settings.
- Tokens arrive one behind the model: the newest is held until the next shows
  it was not the stop token.

The package depends on the PicoMLX forks of mlx-swift and mlx-swift-lm at the
branch `pico/integration`. SwiftPM wants one branch requirement for a package
across the whole graph, so an app that also depends on those forks has to name
the same branch.

MLX needs its Metal library, which only an Xcode build produces, so the live
test runs with `xcodebuild`, not `swift test`:

```sh
TEST_RUNNER_ENGINEROUTER_MLX_MODEL_DIR=/path/to/model \
xcodebuild test -scheme EngineRouter-Package -destination 'platform=macOS' \
  -only-testing:EngineRouterTests/MLXLiveTests
```

## Building the Local Code library

`EngineRouterLocalCode` loads `PicoLocalCodeNative.dylib`, which you build
from the `engine-router` branch of the fork with Xcode and
[Bazelisk](https://github.com/bazelbuild/bazelisk) on Apple silicon:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone --branch engine-router https://github.com/PicoMLX/local-code-community.git
cd local-code-community
ln -s WORKSPACE.darwin WORKSPACE
echo 'try-import %workspace%/.bazelrc.darwin' > .bazelrc
bazelisk build -c opt //Apps/LocalCode:PicoLocalCodeNative
```

The library is `bazel-bin/Apps/LocalCode/PicoLocalCodeNative.dylib`. It
requires macOS 27.

A model directory holds the `.ckpt` checkpoint, its `.engram` file with the
same base name, and optionally the DSpark drafter (a second `.ckpt` whose name
contains `dspark`). To run the live test against it:

```sh
ENGINEROUTER_LOCAL_CODE_LIBRARY=/path/to/PicoLocalCodeNative.dylib \
ENGINEROUTER_LOCAL_CODE_MODEL_DIR=/path/to/deepseek \
swift test --filter LocalCodeLiveTests
```

Its models are chat models: Local Code's own code lays out DeepSeek 4.1's
prompts, tools included, and parses its replies. A model keeps the KV caches
of its last four conversations, so a turn that goes on from an earlier reply
evaluates only its new tokens; `trim(toFree:)` lets those caches go. Through
the token-level interface, each request starts from an empty KV cache.
Requests run one at a time, and a `temperature` other than 0 samples with the
model's own settings.

## Building the TensorFold library

`EngineRouterTensorFold` loads `libPicoTensorFold.dylib`, which you build from
the `engine-router` branch of the fork with Xcode and
[Zig 0.17.0](https://ziglang.org/download/), the exact version TensorFold pins:

```sh
git clone --branch engine-router https://github.com/PicoMLX/TensorFold.git
cd TensorFold
zig build pico-lib -Dcpu=apple_m1
```

The library is `zig-out/native/lib/libPicoTensorFold.dylib`. It needs no other
files at run time. Its minimum macOS version is the version of the Mac that
built it.

The fork's `AGENTS.md` comes from upstream and tells coding agents to register
on a public tracker. Nothing in this package needs that.

```swift
import EngineRouterTensorFold

let router = Router(
    engines: [try TensorFoldEngine(libraryURL: libraryURL)],
    memoryBudget: 96 << 30)
```

- TensorFold talks to Metal directly and links no MLX, so it can be loaded
  beside the MLX adapter. It budgets its memory against the whole process,
  so the engine is exclusive: the router opens its model only when nothing
  else is open.
- Requests to one model run together in the engine's lanes. A request that
  sets no `temperature`, or sets 0, picks the most likely token.
- `canServe` checks the model family and format, not every geometry, so
  `open` can still refuse a model that `canServe` accepted. TensorFold's
  Metal backend runs specific checkpoints of each family; for example it
  refuses `mlx-community/Qwen3.5-4B-bf16`.
- `memory` is the whole process's footprint, and there is nothing to `trim`.

To run the live test:

```sh
ENGINEROUTER_TENSORFOLD_LIBRARY=/path/to/libPicoTensorFold.dylib \
ENGINEROUTER_TENSORFOLD_MODEL_DIR=/path/to/model \
swift test --filter TensorFoldLiveTests
```

So far that test has only run against a synthetic checkpoint with random
weights in a supported geometry, which exercises every call but says nothing
about output quality.

## Building the mlx-serve library

`EngineRouterMLXServe` loads `libPicoMlxServe.dylib`, which you build from the
`engine-router` branch of the fork with Xcode,
[Zig 0.17.0](https://ziglang.org/download/) and CMake:

```sh
git clone --branch engine-router https://github.com/PicoMLX/mlx-serve.git
cd mlx-serve
git submodule update --init --depth 1 lib/mlx-src lib/mlxc-src
./scripts/build-mlx-static.sh
zig build pico-lib -Dpico-only -Doptimize=ReleaseFast
```

The library is `zig-out/lib/libPicoMlxServe.dylib`, and `mlx.metallib` has to
stay beside it. It requires macOS 26.2.

```swift
import EngineRouterMLXServe

let router = Router(
    engines: [try MLXServeEngine(libraryURL: libraryURL)],
    memoryBudget: 96 << 30)
```

- The library carries its own copy of MLX, linked statically with every
  symbol hidden, so it runs in a process that also uses Swift MLX. Each copy
  keeps its own memory accounting.
- It opens one model per process and does its own memory admission, so the
  engine is exclusive. While a model is open, its copy of MLX has a wired
  limit of the machine's recommended working set and a buffer cache limit of
  8 GB.
- Only the MLX engine is in the library: no llama.cpp, ds4, Neural Engine,
  images or outside plugins. The chat template is not loaded, as the app
  renders prompts.
- For bf16 weights mlx-serve decodes with 8-bit copies of the attention
  projections, so its tokens can differ from another engine's for the same
  model.
- Requests to one model run together in the engine's scheduler, and it keeps
  prompt prefixes, so a repeated prompt reports cached prompt tokens. `trim`
  drops those prefixes and its buffer pool.
- mlx-serve ends the process if its inference thread is stuck for 30 seconds
  while a model closes.

To run the live tests:

```sh
ENGINEROUTER_MLXSERVE_LIBRARY=/path/to/libPicoMlxServe.dylib \
ENGINEROUTER_MLXSERVE_MODEL_DIR=/path/to/model \
swift test --filter MLXServeLiveTests
```

`MLXCoexistenceTests` runs this engine and the Swift MLX engine on one model
at the same time. It uses Swift MLX, so it runs with `xcodebuild` and takes
`TEST_RUNNER_ENGINEROUTER_MLXSERVE_LIBRARY` and
`TEST_RUNNER_ENGINEROUTER_MLX_MODEL_DIR`.

## Running Osaurus's engine

The Osaurus app's engine is a separate library,
[vmlx-swift](https://github.com/osaurus-ai/vmlx-swift): batching across
requests, a prompt-prefix cache and its own model formats on its own fork of
MLX. It ships modules with the same names as the MLX libraries this package
links (`MLX`, `MLXLLM`, `MLXLMCommon` and others), so the two cannot be built
into one process. `EngineRouterVMLX` therefore runs it in a helper: an
executable it starts and talks to over standard input and output, or, inside
the App Store sandbox, an XPC service in the app's bundle. The messages are
the same either way.

The helper is its own package, in `Helpers/VMLX`, and depends on the
`engine-router` branch of our fork. Build it with Xcode's build system, which
also builds MLX's Metal library:

```sh
cd Helpers/VMLX
xcodebuild build -scheme enginerouter-vmlx-helper -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath .build/xcode \
  -skipMacroValidation -skipPackagePluginValidation
```

The executable is `.build/xcode/Build/Products/Release/enginerouter-vmlx-helper`.
`mlx-swift_Cmlx.bundle` from the same folder has to stay beside it.

```swift
import EngineRouterVMLX

let router = Router(
    engines: [try await VMLXEngine(helperURL: helperURL)],
    memoryBudget: 96 << 30)
```

- The helper starts with the engine and exits when the engine and its models
  are released. If it dies, every request fails and the engine cannot be used
  again; nothing restarts it.
- **Starting a process is not allowed inside the App Store sandbox.** A
  sandboxed app runs the helper as an XPC service instead; see below.
- A model's weights are in the helper's memory, not the app's. `memory`
  reports the helper's MLX figures, which all models open in it share, and
  `loadBytes` is the size of the weight files.
- Requests to one model run together in vmlx-swift's batch engine. Its
  in-memory prompt-prefix cache is on, so a repeated prompt reports cached
  prompt tokens; its disk cache is off.
- A request that sets no `temperature` or `topP` uses the model's
  `generation_config.json`, or vmlx-swift's defaults when it has none.
- Only text models are served.

### As an XPC service, in the App Store sandbox

The helper package also has a library, `EngineRouterVMLXService`, for an XPC
Service target in the app's own Xcode project:

1. Add a target from Xcode's macOS XPC Service template and replace its
   sources with one `main.swift`:

   ```swift
   import EngineRouterVMLXService

   VMLXHelper.runXPCService()
   ```

2. Add `Helpers/VMLX` to the project as a local package and link
   `EngineRouterVMLXService` to the service target. It is a package of its
   own inside this repository, so it cannot be added by URL. Xcode builds
   MLX's Metal library and copies `mlx-swift_Cmlx.bundle` into the service's
   `Contents/Resources`, where MLX looks for it.
3. Keep the template's Embed XPC Services phase, which puts the service in
   the app's `Contents/XPCServices`.
4. Give the service the App Sandbox entitlement. It reads model folders
   itself, under its own sandbox: a folder the user chose in the app is not
   open to it. Keep models where both can read them, such as an app group
   container that the app and the service both name in
   `com.apple.security.application-groups`.

```swift
import EngineRouterVMLX

let engine = try await VMLXEngine(xpcServiceName: "com.example.app.vmlx-helper")
```

- `xpcServiceName` is the service's bundle identifier. A service the bundle
  does not have is a `VMLXError.launch` error.
- Each engine gets a session of its own: the app hands the service one end of
  each of two pipes, and the messages go over them as over standard input and
  output. Releasing the engine and its models ends the session, which closes
  its models. If the service dies, every request fails as when the helper
  process exits.
- Sessions share the service's process, and with it the MLX figures `memory`
  reports. Their model loads take turns, since a load changes process-wide
  MLX limits. The process stays between sessions; a model's memory goes back
  as it closes.

To run the live tests:

```sh
ENGINEROUTER_VMLX_HELPER=/path/to/enginerouter-vmlx-helper \
ENGINEROUTER_VMLX_MODEL_DIR=/path/to/model \
swift test --filter VMLXLiveTests
```

## Licence note

EngineRouter has no licence file yet; until one is added, all rights are
reserved.

**Local Code is licensed under GPL-3.0**, and that affects how the Local Code
adapter can be used:

- This repository contains no Local Code source and ships no Local Code
  binary. The adapter loads a dynamic library that you build yourself from the
  fork.
- The Free Software Foundation treats dynamically linking a GPL library as
  forming a combined work
  ([GPL FAQ](https://www.gnu.org/licenses/gpl-faq.html#GPLStaticVsDynamic)).
  An app distributed with the Local Code library is therefore subject to
  GPL-3.0 as a whole.
- GPL-3.0 is widely considered incompatible with the App Store's terms, for
  open-source apps as well as closed-source ones.

Until Draw Things grants different terms, treat the Local Code adapter as
**for local developer builds only**: do not ship it in a distributed app, and
do not ship it in a Mac App Store or iOS App Store build. This is a working
rule, not legal advice.

The other engines listed above are under permissive licences (MIT or
Apache-2.0). mlx-serve's distribution also bundles third-party components
with their own notices, which need a review before it ships.
