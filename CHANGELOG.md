# Changelog

All notable changes to GeometricMachineLearning.jl are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0, so a minor bump is a
breaking release).

> [!NOTE]
> Entries for 0.1.0 through 0.4.8 were reconstructed from git history, the release tags and the
> merged pull requests, not written at the time. They are accurate about *what* changed and are
> deliberately coarser about detail than the [Unreleased] section below, which was written
> alongside the work. Where a release removed exported names the list is given; where it is a
> reconstruction of intent, it says so.

## [Unreleased] — targeting 0.5.0

**The optimizer machinery moves to [GeometricOptimizers][go].** GML no longer implements its own
optimizers: the methods, caches, states, global sections and retractions all come from
GeometricOptimizers v0.2, and GML keeps only the parts that are about neural networks — walking a
`NeuralNetworkParameters` tree, and the manifold layer types.

This is a breaking release and the break is not mechanical. Read *Changed (breaking)* before
upgrading.

[go]: https://github.com/JuliaGNI/GeometricOptimizers.jl
[go45]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/45

### Removed (breaking)

- **`BFGSOptimizer` and `BFGSCache`.** GeometricOptimizers has `_BFGS()` and its own cache. GML's
  copies were a replication of it and are gone, along with `docs/src/optimizers/bfgs_optimizer.md`.
- **`SymplecticStiefelManifold`.** Never reachable — the file that defined it was commented out of
  the module.
- **`default_optimizer`.** The optimizer is now always given explicitly.
- **`𝔄` and `𝔄exp`**, and `src/optimizers/manifold_related/modified_exponential.jl` with them.
  `𝔄` was already GeometricOptimizers'; `𝔄exp` moved there in
  [GeometricOptimizers#45](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/45), where it now
  defaults to `ScaledSquaring()` rather than the unscaled Taylor series. Neither was exported.

`BFGSOptimizer`, `BFGSCache`, `SymplecticStiefelManifold` and `default_optimizer` are the whole of
the change to the exported surface, checked against `names(GeometricMachineLearning)` rather than by
reading the export list — the list spans continuation lines, and reading it misses them.

### Changed (breaking)

- **The optimizer constructor takes the method first, and the step size separately.** The learning
  rate is no longer part of the method:

  ```julia
  o = Optimizer(nn, AdamOptimizer(1e-1))              # before
  o = Optimizer(Adam(Float64), nn; step_size = 1e-1)  # after
  ```

  The old signatures were deliberately not kept as deprecations: optimizer functionality belongs in
  GeometricOptimizers, and a compatibility layer here would have had to be removed again.

- **The method types are GeometricOptimizers'.** `GradientMethod`, `MomentumMethod`, `Adam` and
  their caches and states are re-exported from there. `GradientOptimizer`, `MomentumOptimizer` and
  `AdamOptimizer` survive as aliases for the first three.

- **`Adam` is constructed with the element type of the parameters**, e.g. `Adam(Float32)`. Passing a
  mismatched type is caught with a message naming what to pass instead.

- **Julia 1.10 is the minimum** (`julia = "1.9"` → `"1.10"`), inherited from GeometricOptimizers.
  1.9 was never satisfiable with this dependency set in practice.

- **`GeometricIntegrators` gains a `[compat]` bound of `0.18.2`.** It is a test-only dependency and
  had none, which let the resolver pick a version whose `SimpleSolvers` requirement conflicts with
  GeometricOptimizers' — a confusing resolver tree instead of a clear "no such version yet".

### Fixed

- **The optimizer path no longer takes ten hours to compile through a function.** Inference spun in
  method-table intersection whenever `GeometricOptimizers.update!` was reached through GML's
  optimizer tree, and it produced no error — CI showed jobs running past 1 h 15 min against ~25–48
  min on `main`, and one cancelled at 6 h. A run of the repro left going to completion took
  **10 h 11 min**; the same work is ~14.5 s now.

  The cause was upstream and is fixed in [GeometricOptimizers#45][go45]: the optimizer cache and
  state structs bounded their type parameters by the `OptimizerSolution`,
  `GradientArrayOrNamedTuple` and `GlobalSectionSingleOrNamedTuple` aliases, which tied all four
  parameters to one `T` underneath nested `Vararg` unions, so every method-table intersection
  re-solved that constraint system in `subtype_unionall`. This release requires the version that
  carries the fix. **It is a user-facing bug, not only a CI one**: the workload is fast at the REPL,
  where every intermediate is concrete, and pathological for anyone who wraps training in a
  function.


- **GML precompiles on Julia 1.10 again.** `using GeometricOptimizers` was a blanket import, and
  GeometricOptimizers exports about twenty names GML defines itself (`Manifold`, `StiefelManifold`,
  `SkewSymMatrix`, `Optimizer`, `rgrad`, …). Julia 1.12's binding partitions tolerate redefining an
  imported binding; 1.10 raises `cannot assign a value to imported variable
  GeometricOptimizers.Manifold` and does not precompile at all. Replaced with `import
  GeometricOptimizers` and an explicit `using GeometricOptimizers: …` list.

- **The Riemannian gradient is applied on manifold parameters again.** `include("utils.jl")` ran at
  line 71 and `include("manifolds/abstract_manifold.jl")` at line 142, so `Manifold` in

  ```julia
  _gml_rgrad(x::Manifold, dp) = rgrad(x, dp)
  ```

  resolved to **`GeometricOptimizers.Manifold`**. GML's `StiefelManifold` does not subtype that, so
  the method never matched: the `_gml_rgrad(x, dp) = dp` fallback caught it and the raw Euclidean
  gradient was passed through unprojected. The optimizer machinery moved out of `utils.jl` into
  `src/optimizers/optimizer.jl`, included after the manifolds. This is the same type split as
  [#234](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/234), showing up somewhere
  it silently changed results.

- **Adam's bias correction was wrong on the Euclidean path, and stopped training almost
  immediately.** The first moment coefficient was written `β/(1 - βᵗ)` where it should be
  `(β - βᵗ)/(1 - βᵗ)`. At `t = 2` that is 49.7 instead of 0.497 — a factor of 100 — and it compounds
  every step, inflating the second moment until `m₁ / (√m₂ + δ)` is indistinguishable from zero.

  Only `AdamOptimizerWithDecay` showed it. Plain `Adam` is one of the methods GeometricOptimizers
  owns, so it takes the upstream cache and update; the decaying variant is not, so it falls back to
  GML's own `_euclidean_update!`, which carried the error. On the regression test the decaying
  optimizer plateaued at 0.386 from epoch 5 while plain Adam reached 0.115. The trajectory now
  matches `main` to five significant figures at every epoch sampled.

  Found by the pre-push hook running the full suite — the first time it has ever run to this point,
  because the compile-time stall used to hang it three groups earlier.

- **The transformer tests no longer import GeometricOptimizers wholesale.** Three
  `multi_head_attention_stiefel_*` files opened with `using GeometricMachineLearning,
  GeometricOptimizers` and then used `StiefelManifold` unqualified, which is ambiguous on Julia
  1.12. `Pkg.test` puts the tested package's dependencies in the test environment, so this failed in
  CI too. It had never been seen because the compile-time stall above hung the suite in the
  reduced-order-modeling group, which runs first.

- **Four wrong assertions in the `batch.jl` doctests.** The doctests were rewritten on this branch
  to assert rather than print, and three of the assertions compared a `Tuple` against a `Vector`
  (`length.(batches) == [2, 2, 1]`, where the value is `(2, 2, 1)` — never equal in Julia), while a
  fourth expected five index pairs from a time-series `DataLoader` that yields four. They failed the
  Documentation and PDF workflows, which run `make test_docs`; `makedocs` itself has
  `doctest = false`, so nothing else would have caught them.

- **Seven unresolvable documentation cross-references.** With no `@autodocs` block anywhere, an
  `@ref` resolves only if some `@docs` entry documents that binding. `GeometricMachineLearning.𝔄`,
  `cayley`, `cayley(::StiefelLieAlgHorMatrix)`, `cayley(::StiefelManifold, ::AbstractMatrix)`,
  `Adam`, `MomentumMethod` and `update!` had none; `AdamOptimizerWithDecay` lost its `@docs` entry
  when `optimizer_methods.md` was rewritten into prose, dangling two tutorial references.

- **Duplicate BibTeX keys** `Kraus:2020:GeometricIntegrators` and `greydanus2019hamiltonian`, which
  had been failing the Documentation workflow on `main` since 2026-07-21.

- **Doctests no longer abort the test suite.** `src/data_loader/batch.jl`'s doctests hardcode
  `shuffle` output and the RNG stream changed in 1.13. The doctest testset is removed from
  `test/runtests.jl` — the docs workflow owns doctest validation, and the examples' behaviour is
  covered by structural assertions under `test/`.

### Added

- **`test/runtests.jl` emits seven `@info` markers**, one per testset group, so that a long job can
  be told from a hung one.
- **Generated-artifact patterns in `.gitignore`**, including `docs/src/tutorials/mnist/*.pdf`. The
  existing rules covered the `.aux` and `.log` siblings but not the compiled PDFs.

### Dependencies

- `GeometricOptimizers` added; the `[sources]` entry that pointed at its `main` branch is gone now
  that v0.2.0 is registered, which also makes GML registrable again.
- `SimpleSolvers` and `ParameterHandling` arrive through GeometricOptimizers but are not referenced
  under `src/`, `ext/` or `test/`, so they get no `[compat]` entries of their own.

## [0.4.8] — 2026-07-07

- HDF5 becomes a weak dependency with an `HDF5Ext` extension, and saving and loading is extended to
  GML's special array types (`SkewSymMatrix`, `SymmetricMatrix`, the triangular and
  Lie-algebra-horizontal types). The tutorials move off JLD2 for parameter storage.
- The symplectic autoencoder gains a highly nonlinear decoder.

## [0.4.7] — 2026-04-21

Dependency updates, a corrected proof for the volume-preserving transformer, and documentation
reference fixes.

## [0.4.6] — 2025-12-10

Typo fix in the symplectic attention documentation.

## [0.4.5] — 2025-07-09

`GeometricIntegrators` is removed as a hard dependency, remaining only as a test and documentation
dependency.

## [0.4.4] — 2025-06-05

Dependency compatibility updates.

## [0.4.3] — 2025-06-04

`FeedForwardLoss` is imported from `AbstractNeuralNetworks` rather than defined here.

## [0.4.2] — 2025-05-14

Symbolic and Hamiltonian neural network work, via `SymbolicNeuralNetworks`.

**Breaking.** `HamiltonianNeuralNetwork` and `tstep` stopped being exported somewhere in the 0.4.x
series; this is the release whose changes concern them. Semantic versioning would have called for a
minor bump.

## [0.4.1] — 2025-02-19

The symplectic transformer architecture, dark-mode figure handling and a simplified documentation
workflow.

## [0.4.0] — 2025-02-14

**Breaking.** These names stopped being exported: `Classification`, `HNNProblem`, `LNNProblem`,
`NeuralNetMethod`, `RegularTransformerIntegrator`, `SympNetMethod`, `SymplecticPotential`,
`accuracy`, `integrate`, `integrate_step!` and `method`. Some were deleted outright; others
(`accuracy`, `integrate`) survive as internals, so code reaching them through the export broke while
`GeometricMachineLearning.accuracy` still works.

- `GeometricIntegrators` and `Documenter` leave `[deps]`.
- The linear symplectic transformer and the standard transformer integrator are added.
- An abstract pullback interface; the loss gradient becomes an optional input argument.
- Several `if` chains are replaced by dispatch.
- Documentation is substantially reworked (arrays, manifolds, optimizers).

## [0.3.0] — 2024-05-07

**Breaking.** These names stopped being exported: `Attention`, `Iterate_Sympnet`,
`ReducedSystem`, `compute_projection_error`, `compute_reduction_error`, `nn`,
`perform_integration_full`, `perform_integration_reduced` and
`reduced_vector_field_from_full_explicit_vector_field`.

- Volume-preserving feedforward networks and the volume-preserving transformer.
- Neural network integrators, the transformer integrator, PSD, the symplectic autoencoder and
  ResNet as first-class architectures.
- Adam with learning-rate decay.
- Neural network loss types, replacing the earlier loss routines.
- Adjusted to a new `AbstractNeuralNetworks` interface.

## [0.2.0] — 2024-01-12

- Recurrent and LSTM architectures, and the transformer neural network.
- The BFGS optimizer.
- SympNet layers are unified; retractions are simplified.
- Custom neural network types are removed in favour of `AbstractNeuralNetworks`, and several
  dependencies are dropped.
- MNIST and symplectic-autoencoder scripts; substantially expanded documentation and test coverage.

## [0.1.0] — 2023-08-15

First release: manifold types (`StiefelManifold`, `GrassmannManifold`), the structured array types
(`SkewSymMatrix`, `SymmetricMatrix`, the triangular and Lie-algebra-horizontal families), SympNets,
manifold layers, multi-head attention, the data loader, and Riemannian optimizers with geodesic and
Cayley retractions.

---

## Open Issues

Everything below came up while getting CI green on
[#230](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/230) and is **not** fixed. Each
entry says what closing it would take.

### A. Blocked on upstream releases

Nothing in this package can go green until these land. The critical path runs entirely outside it.

- **A1. `GeometricIntegrators 0.18.2` is not released.** GeometricOptimizers requires
  `SimpleSolvers = "0.12"`; registered `GeometricIntegrators 0.18.1` requires `"0.11"`. There is no
  overlap, and `GeometricIntegrators` is in both the test target and `docs/Project.toml`, so *every*
  job dies at resolution in about 40 seconds — all twelve CI jobs plus Documentation and PDF, the
  latter two on `make test_docs` with `Unsatisfiable requirements detected for package
  SimpleSolvers`.

  SimpleSolvers 0.12.1 and GeometricIntegratorsBase 0.6.3 are registered. This is the last one.

- ~~**A2. GeometricOptimizers 0.2.1 is not released.**~~ **Closed.** v0.2.1 is registered, and the
  bound here is `GeometricOptimizers = "0.2.1"` — a floor rather than tidiness, since `"0.2"` lets
  the resolver pick 0.2.0 and silently reinstate a ten-hour compile, which presents as a job that
  outlasts its timeout rather than one that fails. The compile-time fix is confirmed against the
  registered version, resolved with nothing developed locally: 13.9 s cold, 6.5 ms warm.

### B. Known defects

- **B1. Both packages export `AdamOptimizerWithDecay`.** GeometricOptimizers v0.2.0 ships the name
  and GML defines and exports its own (`src/optimizers/optimizer.jl:35`). GML itself loads, but
  `using GeometricMachineLearning, GeometricOptimizers` in downstream code fails outright:

  ```
  UndefVarError: `AdamOptimizerWithDecay` not defined in `Main`
  Hint: It looks like two or more modules export different bindings with this name…
  ```

  Deleting GML's copy is the fix, but it is **not** independent of C1: GeometricOptimizers'
  `AdamOptimizerWithDecay(n, T; …)` returns an `(algorithm, linesearch)` pairing for its own
  `Optimizer(x, problem; method...)`, whereas GML's `Optimizer` carries a scalar `step_size` and
  computes the schedule in `_current_step_size`. Un-exporting it would clear the ambiguity on its
  own if a stopgap is wanted first.

- **B2. `Manifold` is split between the two packages
  ([#234](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/234)).** GML's
  `StiefelManifold` and `GrassmannManifold` subtype `GeometricMachineLearning.Manifold`, which is
  distinct from `GeometricOptimizers.Manifold`, so GeometricOptimizers' generic `geodesic` and
  `cayley` never dispatch on them. Commit `b16267ea` worked around this by re-implementing the
  pipeline four times, with the same duplication on the Lie-algebra-horizontal types and the
  `_copyto!`/`_add!`/`_rac!`/`_square!`/`_div!` family — about thirty bridge methods.

  The decided fix is `const Manifold = GeometricOptimizers.Manifold`, which lets all four bridge
  methods be deleted. It is not a one-liner: `src/manifolds/abstract_manifold.jl` is a near-verbatim
  copy of GeometricOptimizers', so after aliasing, GML's generic methods would have *identical*
  signatures to the upstream ones and silently overwrite them — a hard precompilation error on Julia
  ≥ 1.13. GML's copies have to go in the same change, keeping only what genuinely differs.

  The `_gml_rgrad` bug fixed in this release was the same split showing up somewhere it changed
  results silently, which is an argument for closing this sooner rather than later.

- **B3. `input_dimension` and `output_dimension` exist twice.** AbstractNeuralNetworks v0.6.4 defines
  and exports them on `AbstractLayer`; SymbolicNeuralNetworks 0.3 defines them by pirating
  AbstractNeuralNetworks' types, and GML imports SymbolicNeuralNetworks'. They are two distinct
  generic functions with the same name and semantics.

  Not a blocker and no `[compat]` change is needed — the explicit `import SymbolicNeuralNetworks: …`
  beats the implicit binding, and GML re-exports neither. Closing it needs
  SymbolicNeuralNetworks to drop its pirated methods and GML to import from AbstractNeuralNetworks,
  which is not a straight swap: AbstractNeuralNetworks 0.6.4 has no `Chain` method and GML relies on
  SymbolicNeuralNetworks' at `src/architectures/hamiltonian_neural_network.jl:86`, so
  `input_dimension(::Chain)` has to exist upstream first.

### C. Follow-ups and cleanups

- **C1. The rest of the optimizer machinery still belongs upstream.** The traversal
  (`_make_optimizer_cache`, `_make_optimizer_state`, `_tree_optim_step!`, `_leaf_optim_step!`) and
  the bespoke `GMLEuclideanState` are GML implementations of what GeometricOptimizers v0.2 already
  supports natively. Route: branch GeometricOptimizers, move it, delete it here, open an issue
  referencing the upstream PR. B1 unblocks with it.

- **C2. Two `isa` branches remain in `_leaf_optim_step!`** (`src/optimizers/optimizer.jl:182`,
  `:187`, for `AdamState`/`MomentumState`). Measurement showed the traversal is not implicated in
  the compile-time problem, so this is tidying, and it disappears entirely if C1 lands first.

- **C3. Cross-package documentation links are prose, not links.** The seven `@ref`s fixed above were
  de-referenced into plain code spans, which is the cheap fix rather than the right one.
  `DocumenterInterLinks` would let GML's documentation link into GeometricOptimizers' properly, so
  that `𝔄`, `cayley` and `update!` become real cross-references again. GeometricOptimizers' own
  `docs/Project.toml` already carries it; GML's does not.

- **C4. `[compat]` entries worth revisiting.** `ForwardDiff = "0.10, 1"` is dead weight —
  GeometricOptimizers requires 1, so the resolver picks it regardless and the `0.10` branch is
  untestable. `LazyArrays = "=2.3.2"` is an exact pin that currently resolves; it is the first thing
  to relax if resolution fails.

- **C5. Julia 1.13 and nightly are marked `experimental: true`.** 1.13 should be de-experimentalised
  once it is green, or an upstream issue linked. Nightly stays experimental permanently, as in the
  other JuliaGNI repositories. Before opening anything, check `RungeKutta` and `GeometricIntegrators`
  for an existing `GenericLinearAlgebra` report to reference.

- **C6. Three generated MNIST PDFs are in this branch's history** for three commits, from a
  `git add docs/src` that swept them in. They are untracked again and `.gitignore` now covers the
  pattern, so the working tree and the net diff are clean, but the blobs are still reachable.
  Rewriting the branch would remove them.

### D. Unverified

Not defects — claims this release makes that nothing has actually checked yet.

- ~~**D1. No full test suite has ever been observed end to end.**~~ **Closed.** The suite now runs
  to completion, all seven `@info` markers, with no failures — on Julia 1.13.0-rc2, via the pre-push
  hook. Getting there took four rounds, each uncovering a bug the previous one had been hiding:
  `_gml_rgrad` skipping the Riemannian projection (found on the 1.10 precompile), the transformer
  tests importing both modules (found once the reduced-order group cleared), Adam's bias correction
  (found once the optimizer group was reached) and the docstring tests' missing import (found once
  the final group was reached).

  Worth keeping the moral: the four bugs were not related to each other, and none was visible until
  the one before it was cleared. A suite that stops at the first failure reports one problem at a
  time however many there are.

- **D2. The test suite has never been run on Julia 1.10.** The package precompiles and loads there
  now, and resolves from the registry, but no suite has run there — and `julia = "1.10"` is a claim
  this release newly makes. The full run above was on 1.13.0-rc2. CI covers 1.10 and is the first
  thing that will exercise it.

- **D3. The Documentation and PDF workflows are unverified.** Both die at resolution long before
  Documenter starts (A1), so the documentation fixes above rest on a static audit of `@ref` targets
  against `@docs` entries. That catches missing entries and says nothing about executable blocks,
  doctests or tutorials. `docs/Makefile`'s `test_docs` target is likewise unrun.

- **D4. The upstream fix was measured on one optimizer.** The compile-time figures come from the
  `Adam` path. The quasi-Newton and Newton caches and states were widened on the strength of their
  *inferred types* — a sound argument, but not a measurement. Catalogued upstream as GeometricOptimizers C15.
