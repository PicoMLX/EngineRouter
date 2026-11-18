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

Early. This repository currently holds the package skeleton and this note.
The engine interface, the router and the first adapters land in the pull
requests that follow.

## Engines

| Engine | How it is sourced | Licence | State |
|---|---|---|---|
| [mlx-swift-lm](https://github.com/ronaldmannak/mlx-swift-lm) | Our fork, on the same branch requirement PicoCore uses | MIT | Planned (`EngineRouterMLX`) |
| [Local Code](https://github.com/drawthingsai/local-code-community) | Our fork, built locally into a dynamic library | GPL-3.0 | Planned (`EngineRouterLocalCode`); see the licence note |
| [mlx-serve](https://github.com/ddalcu/mlx-serve) | Our fork, exporting its existing C engine table from a library | MIT | Planned |
| [Sushi](https://github.com/beamivalice/sushi) EXL3 | Through mlx-serve's vendored EXL3 module | Via mlx-serve | Planned |
| [TensorFold](https://github.com/ashhart/TensorFold) | A fork, when that adapter starts | Apache-2.0 | Later |

Each adapter is its own library product. An app that depends only on
`EngineRouter` links no engine.

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
