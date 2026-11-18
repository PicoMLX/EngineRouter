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

Early. The engine interface, the router, and the MLX, Local Code and
TensorFold adapters exist. The mlx-serve adapter is next.

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

A character can span several tokens, and half of one decodes to nothing or to
U+FFFD, so decode tokens together, never one event at a time. To show text as
it arrives, decode the reply so far and emit what that adds to the text
already shown.

The interface is token-level: the app renders the chat template and parses
tool calls, so both behave the same on every engine. Cancelling the task that
consumes a request's stream cancels the request.

`Router.open` goes to the first engine whose `canServe` accepts the model and
charges the model's `loadBytes` against the budget. It throws when the model
does not fit or when an exclusive engine is involved; the router never closes
a model on its own, so the caller decides what to close.

## Engines

| Engine | How it is sourced | Licence | State |
|---|---|---|---|
| [mlx-swift-lm](https://github.com/PicoMLX/mlx-swift-lm) | Our fork, on the same branch requirement PicoCore uses | MIT | `EngineRouterMLX`, text only, one request at a time per call; see below |
| [Local Code](https://github.com/drawthingsai/local-code-community) | [Our fork](https://github.com/PicoMLX/local-code-community/tree/pico/native-bridge), built locally into a dynamic library | GPL-3.0 | `EngineRouterLocalCode`, macOS only, DeepSeek 4.1 checkpoints; see the licence note |
| [mlx-serve](https://github.com/ddalcu/mlx-serve) | Our fork, exporting its existing C engine table from a library | MIT | Planned |
| [Sushi](https://github.com/beamivalice/sushi) EXL3 | Through mlx-serve's vendored EXL3 module | Via mlx-serve | Planned |
| [TensorFold](https://github.com/ashhart/TensorFold) | [Our fork](https://github.com/PicoMLX/TensorFold/tree/pico/native-bridge), built locally into a dynamic library | Apache-2.0 | `EngineRouterTensorFold`, macOS only; so far run only against a synthetic checkpoint, see below |

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
from the `pico/native-bridge` branch of the fork with Xcode and
[Bazelisk](https://github.com/bazelbuild/bazelisk) on Apple silicon:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone --branch pico/native-bridge https://github.com/PicoMLX/local-code-community.git
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

Each request starts from an empty KV cache, requests run one at a time, and a
`temperature` other than 0 samples with the model's own settings.

## Building the TensorFold library

`EngineRouterTensorFold` loads `libPicoTensorFold.dylib`, which you build from
the `pico/native-bridge` branch of the fork with Xcode and
[Zig 0.17.0](https://ziglang.org/download/), the exact version TensorFold pins:

```sh
git clone --branch pico/native-bridge https://github.com/PicoMLX/TensorFold.git
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
