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

- **`BFGSOptimizer` and `BFGSCache`**, along with `docs/src/optimizers/bfgs_optimizer.md`.

  This entry used to say that GeometricOptimizers "has `_BFGS()` and its own cache" and that GML's
  copies "were a replication of it". **That is wrong, and BFGS training of a neural network is
  currently lost rather than relocated.** The two are different algorithms:

  | | GML's `BFGSOptimizer(η, δ)` | GeometricOptimizers' `BFGS` |
  |---|---|---|
  | driven by | `optimization_step!`, gradient only | a cache holding an inverse-Hessian approximation |
  | step | fixed learning rate `η` | quasi-Newton direction, `Q` sized by the *flattened* parameters |
  | fits GML's per-leaf tree update? | yes — that is what it was for | no |

  `_is_go_native_method` therefore sends `BFGS` down GML's Euclidean path, where
  `_euclidean_update!` has no method for it and the step raises a `MethodError`. Bridging it needs
  `_fill!`, `_difference!`, `outer!` and the `ParameterHandling.flatten` round-trip taught about
  GML's manifold and lift types — GML's `StiefelManifold` is a *different type* from
  GeometricOptimizers', and the two hierarchies are unrelated, so none of GO's `Manifold` methods
  apply. That work is not done.

  Until it is, use `AdamOptimizer()`, `MomentumOptimizer()` or `GradientOptimizer()`.
- **`SymplecticStiefelManifold`.** Never reachable — the file that defined it was commented out of
  the module.
- **`default_optimizer`.** The optimizer is now always given explicitly.
- **`𝔄` and `𝔄exp`**, and `src/optimizers/manifold_related/modified_exponential.jl` with them.
  `𝔄` was already GeometricOptimizers'; `𝔄exp` moved there in
  [GeometricOptimizers#45](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/45), where it now
  defaults to `ScaledSquaring()` rather than the unscaled Taylor series. Neither was exported.

- **`split_and_flatten` and `onehotbatch`**, together with `src/data_loader/mnist_utils.jl` and the
  unexported index arithmetic behind them (`patch_index`, `within_patch_index`, `index_conversion`).

- **`DataLoader(::AbstractArray{T, 3}, ::AbstractVector)`**, the image-classification constructor.
  Its own docstring described it as "tailored towards being used with the package MLDatasets.jl",
  which is what made it the cut. Every other `DataLoader` constructor is unchanged.

  All of this moved to **[GMLDatasets](https://github.com/JuliaGNI/GMLDatasets.jl)**, along with the
  MNIST tutorial and the MNIST scripts. GML is a library for scientific machine learning and had no
  business pulling an image-dataset package into its documentation build to document itself — the
  MNIST tutorial *downloaded the data set every time the docs were built*. `MLDatasets` is now
  absent from `docs/Project.toml` and `scripts/Project.toml`, and the docs build is offline again.

  No deprecation shims: GMLDatasets depends on GML, so a forwarding shim here would be a dependency
  cycle. To port a script, add GMLDatasets and change the data-loading lines:

  ```julia
  # before
  using GeometricMachineLearning
  import MLDatasets
  train_x, train_y = MLDatasets.MNIST(split = :train)[:]
  dl = DataLoader(train_x, train_y; patch_length = 7)

  # after
  using GeometricMachineLearning, GMLDatasets
  dl = mnist_data_loader(:train; patch_length = 7)
  ```

  What stayed: `ClassificationTransformer`, `ClassificationLayer`, `ClassificationTransformerLoss`
  and `accuracy`. None of them is specific to image data, and they are what GMLDatasets' tutorial
  trains.

`BFGSOptimizer`, `BFGSCache`, `SymplecticStiefelManifold`, `default_optimizer`, `split_and_flatten`
and `onehotbatch` are the whole of the change to the exported surface, checked against
`names(GeometricMachineLearning)` rather than by reading the export list — the list spans
continuation lines, and reading it misses them.

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

- **Two default hyper-parameters changed with the move to GeometricOptimizers' methods**, and a
  script that relied on the old defaults will train differently:

  | | before | after |
  |---|---|---|
  | momentum step size | `MomentumOptimizer()` → `η = 1e-3` | `_default_step_size` → `1e-2` |
  | Adam's `δ` | `AdamOptimizer()` → `3f-7` | GeometricOptimizers' `Adam` default |

  The gradient method's default step size is unchanged at `1e-2`, and Adam's is unchanged at `1e-3`.
  Pass `step_size` and `δ` explicitly if the old values matter.

- **Julia 1.10 is the minimum** (`julia = "1.9"` → `"1.10"`), inherited from GeometricOptimizers.
  1.9 was never satisfiable with this dependency set in practice.

- **`GeometricIntegrators` gains a `[compat]` bound of `0.18.2`.** It is a test-only dependency and
  had none, which let the resolver pick a version whose `SimpleSolvers` requirement conflicts with
  GeometricOptimizers' — a confusing resolver tree instead of a clear "no such version yet".

- **`TrainingParameters` and `TrainingSet` take the optimizer explicitly**, which is the *Removed*
  entry on `default_optimizer` reaching the constructors that still called it:

  ```julia
  TrainingParameters(nruns, method)          # before — raised UndefVarError
  TrainingParameters(nruns, method, mopt)    # after

  TrainingParameters(nn, data)                                    # before — raised UndefVarError
  TrainingParameters(nn, data, mopt; method = …, nruns = …)       # after

  TrainingSet(es)                            # before
  TrainingSet(es, mopt = GradientOptimizer())  # after
  ```

  The two-argument `TrainingParameters(nn, data)` called `default_optimizer()` *and*
  `default_integrator(nn, data)`, neither of which exists — the second has been `default_method` for
  some time — so it could not be called at all. The optimizer is a required argument there;
  `TrainingSet(::EnsembleSolution)` keeps a default, since it is the constructor that exists to fill
  everything in from a solution.

- **CairoMakie is the only plotting library.** GLMakie is gone from `docs/Project.toml` and the six
  documentation pages that used it, and Plots from `scripts/` and the legacy `hnn/` scripts. Every
  2D figure already used CairoMakie and `docs/src/manifolds/manifolds.md` already rendered a 3D
  scene with it; the six holdouts are static `Axis3` renders with no interactivity or animation, so
  CairoMakie's lack of per-pixel depth resolution does not bite. The one visible difference is that
  GLMakie dimmed arrows on the far side of a semi-transparent sphere and CairoMakie draws them at
  full strength.

  This is what removes `xvfb` from CI: the Documentation and LaTeX workflows no longer install
  `xorg-dev mesa-utils xvfb libgl1 freeglut3-dev libx*` or wrap anything in `xvfb-run`, and
  `docs/gl_makie_transparent_background_hack.jl` — a `colorbuffer` trick for saving a transparent
  background, which CairoMakie does natively — is deleted.

- **`code_generation`, `mt_fun` and `hnn` moved to `legacy/codegen`, `legacy/mtk` and `legacy/hnn`.**
  None is reachable from the package, its tests or its documentation; they sat next to `src/` and
  `scripts/` as though they were current, and `hnn/` predates two generations of the optimizer and
  architecture APIs.

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

- **A docstring index that stopped being a list.** Deleting the BFGS page left
  `value_for_key(_optimizers, "Optimizer Methods")` with a single key, and that method returns a
  `String` where the multi-key one returns a `Vector{String}`. `docstring_index.md` passes the result
  straight to `@index` as `Pages`, so the Documentation and PDF builds died with
  `Cannot convert an object of type String to an object of type Vector{String}`. Wrapped in `[ ]`,
  as the one other single-entry chapter already was.

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

- **The Hamiltonian vector field is built out-of-place.** `SymbolicNeuralNetworks.build_nn_function`
  returns an `InPlaceBatchedFunction` by default in 0.5, and its result is produced by mutation.
  `HNNLoss` wraps that function and is differentiated with `Zygote`, which raises `Mutating arrays
  is not supported`. `hamiltonian_vector_field` now passes `inplace = false`.

- **`SymbolicPullback(::HamiltonianArchitecture)` produced a gradient of the wrong loss.**
  `SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` sizes the symbolic target of the loss with
  `output_dimension(nn.model)`, which for a Hamiltonian network is `1` — the scalar Hamiltonian.
  `HNNLoss` compares against the Hamiltonian *vector field*, whose dimension is that of the network
  *input*. Under SymbolicNeuralNetworks 0.3 the mismatch was hidden by `Symbolics.Arr` broadcasting
  and gave a wrong gradient; 0.5's scalar variables turn it into a `DimensionMismatch`. GML now
  builds the pullback itself with the correct dimension.

- **The symbolic vector field is a vector.** It used to be an ``n\times{}1`` matrix, so evaluating
  it on a single sample returned a matrix where `HNNLoss` compares it against a vector.

- **`test_hnn_loss_derivative` was never called.** It is defined in
  `test/hamiltonian_neural_network_tests.jl` and was never invoked, so nothing exercised the
  `Zygote` gradient of the HNN loss — the one thing that catches the in-place break above. It also
  asserted on a `.params` field that the gradient has not had since AbstractNeuralNetworks 0.6.
  Called and repaired, and joined by a test for `SymbolicPullback(::HamiltonianArchitecture)`, which
  had no coverage at all.

- **`input_dimension` and `output_dimension` are one generic function again.** They used to exist
  twice: AbstractNeuralNetworks defines and exports them on `AbstractLayer`, SymbolicNeuralNetworks
  0.3 defined its own by pirating AbstractNeuralNetworks' types, and GML imported
  SymbolicNeuralNetworks'. 0.5 stopped doing that — it `import`s AbstractNeuralNetworks' and adds
  the `Chain` methods to them — so GML imports both from AbstractNeuralNetworks now. The `Chain`
  methods still live in SymbolicNeuralNetworks and still belong upstream, which is
  [SymbolicNeuralNetworks#35](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/35).

- **The step size can be set through `train!` again.** It moved from the optimizer method onto
  `Optimizer` (see *Changed*), and `train!` kept building `Optimizer(m, nn)` without forwarding one
  — so every run silently took `_default_step_size(m)` and the documented way of choosing a learning
  rate did nothing. There is a `step_size` keyword now, defaulting to what the method asks for.

- **The three-argument `train!` always raised.** Its training method defaulted to
  `default_method(nn, data)` while the parameter is named `_data`, so `data` resolved to GML's
  exported accessor *function* of that name and the call died with a `MethodError` before it began.

- **`test/training_parameters.jl` is new and is in `runtests.jl`.** Nothing caught either of the two
  bugs above, or the `default_optimizer` one under *Removed*, because everything that touches
  `TrainingParameters` lives under `test/train!/`, which `runtests.jl` does not include. The new file
  pins the step size by asserting that `step_size = 0` leaves the loss — which `train!` recomputes
  over the whole data set after every step — identical at every step.

### Added

- **`test/runtests.jl` emits seven `@info` markers**, one per testset group, so that a long job can
  be told from a hung one. The suite runs to completion, all seven markers and no failures, on
  Julia 1.13.0-rc2 — the first time it has been observed end to end. Getting there took four rounds,
  each uncovering a bug the previous one had been hiding (`_gml_rgrad` skipping the Riemannian
  projection, the transformer tests importing both modules, Adam's bias correction, the docstring
  tests' missing import): none of the four was related to the others, and none was visible until the
  one before it was cleared.
- **Generated-artifact patterns in `.gitignore`**, including `docs/src/tutorials/mnist/*.pdf`. The
  existing rules covered the `.aux` and `.log` siblings but not the compiled PDFs.

### Dependencies

- `GeometricOptimizers` added; the `[sources]` entry that pointed at its `main` branch is gone now
  that it is registered, which also makes GML registrable again. The bound is
  `GeometricOptimizers = "0.2.1"` — a floor rather than tidiness, since `"0.2"` lets the resolver
  pick 0.2.0 and silently reinstate the ten-hour compile below, which presents as a job that
  outlasts its timeout rather than one that fails.
- `GeometricIntegrators = "0.18.2"`. Until it was registered, GeometricOptimizers' `SimpleSolvers =
  "0.12"` and `GeometricIntegrators 0.18.1`'s `"0.11"` had no overlap, and since
  `GeometricIntegrators` is in both the test target and `docs/Project.toml`, *every* job died at
  resolution in about 40 seconds.
- `SimpleSolvers` and `ParameterHandling` arrive through GeometricOptimizers but are not referenced
  under `src/`, `ext/` or `test/`, so they get no `[compat]` entries of their own.
- **`SymbolicNeuralNetworks = "0.5"`** (was `"0.3"`). 0.5 is a refactor of the whole package; what
  GML relies on is that `_get_params` and `_get_contents` are gone — they were never about
  symbolics, they clean up what `Zygote` returns, and GML now defines the three methods it uses next
  to `_processing` in `src/pullbacks/zygote_pullback.jl` — and that generated functions are in-place
  by default (see *Fixed*). 0.3 is dropped rather than kept alongside: the two APIs differ in the
  type of `nn.input` as well, so a single source tree cannot serve both, and CI would only ever
  exercise whichever the resolver picks.
- **`AbstractNeuralNetworks = "0.6.4"`** (was `"0.5, 0.6"`). SymbolicNeuralNetworks 0.5 requires
  `"0.6.4"`, so `0.5` could never resolve alongside it.
- **Four dependencies dropped as unused**: `BandedMatrices` (no `BandedMatrix`, no `bandwidth`),
  `SparseArrays` (no occurrence at all — the only matches for `sparse` are the word in two
  docstrings), `StatsBase`, and `ZygoteRules` (no `@adjoint`, no `@nograd`). `BandedMatrices` also
  had a module-level `using` that nothing needed.

  `Distances` looked equally dead and is **not**: `sqeuclidean` is the default distance of every
  `TrainingMethod` in `src/training_method/`, reached through the module-level `using` rather than a
  qualified call. It stays, with a comment saying why, and the test suite is what caught it —
  `test/data/test_batch.jl` failed with `UndefVarError: sqeuclidean` when it went.
- **`Printf`, `ChainRulesTestUtils` and `SafeTestsets` move from `[deps]` to `[extras]`.** All three
  are test-only; the last two were already duplicated in `[extras]`. `Printf` joins the test target
  because `test/custom_ad_rules/kernel_pullbacks.jl` uses it and *is* in `runtests.jl`.
- **`Aqua` and `RungeKutta` dropped from `[extras]`** — they appear nowhere in the repository and
  were in no target — along with `GeometricEquations` and `Random`, which are already in `[deps]`.
  `Documenter` leaves the test target: there is no `doctest` or `DocMeta` anywhere under `test/`, and
  the documentation workflow owns doctest validation.
- **`GLMakie` and `Plots` dropped from `docs/Project.toml`**; `Plots`, `Functors`, `Parameters` and
  `ProfileView` from `scripts/Project.toml`, which also gains the ten packages its own scripts use
  without declaring (`StatsBase`, `Lux`, `ProgressMeter`, `Distances`, `KernelAbstractions`,
  `NLsolve`, `BenchmarkTools`, `OffsetArrays`, `SafeTestsets`, `Test`). Several scripts could not run
  in that environment at all.

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
[#230](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/230) and while moving to
SymbolicNeuralNetworks 0.5 in
[#235](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/235), and is **not** fixed. Each
entry says what closing it would take. Entries that have since been closed are not kept here — what
they resolved to is in the release notes above.

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

- **B4. The documentation's executable examples still use the old optimizer internals.** The
  Documentation and PDF builds get past resolution and doctests now and fail in the `@example`
  blocks — 32 errors across 12 pages, because the documentation teaches the optimizer by reaching
  into its cache, and GeometricOptimizers' caches are not shaped like GML's were:

  ```
  type MomentumCache has no field `A`, available fields: `x`, `g`, `δ`, `Δg`, `g̃`, `g̃_is_current`, `section`
  type AdamCache has no field `Y`
  no method matching update!(::Optimizer{GradientMethod, GradientCache{…}}, …)
  `update_section!` not defined
  no method matching AdamCache(::@NamedTuple{weight::SymmetricMatrix{Float16, …}}, …)
  ```

  `optimizer_methods.md` is the bulk of it: eleven call sites building `dx = (A = one(weight.A),)`,
  calling `update!(o, o.cache, dx)` and printing `o.cache.A` or `o.cache.Y` to show what a cache
  holds. `parallel_transport.md` uses `update_section!`, which GML no longer exports. The three
  named `@example` blocks that fail outright are `sympnet`, `rigid_body` and `s2_parallel_transport`.

  This is not a mechanical rename. The pages explain how the optimizer works by showing its
  internals, and those internals now belong to another package with a different design — so closing
  this means deciding how much of that exposition GML's documentation should still carry, and
  rewriting it against the upstream API or handing it to GeometricOptimizers' own documentation. It
  is the last thing standing between this branch and green Documentation and PDF workflows.

- **B5. The symbolic pullback of `HNNLoss` is not the gradient of the batched loss.**
  `SymbolicPullback` differentiates the loss of a *single* sample and sums the per-sample gradients
  (`reduce = +`), which equals the gradient of the batched loss only when the loss is a sum over
  samples. `HNNLoss` is not: it divides by `norm(output)` taken over the whole batch. Measured on a
  `StandardHamiltonianArchitecture(2, 2)` against a `Zygote` gradient of the same loss:

  ```
  N=1  symbolic=-0.06324019893769198  zygote=-0.06324019893769196  agree=true
  N=5  symbolic=-0.29982310048056143  zygote=-0.05705562794369161  agree=false
  ```

  `SymbolicPullback(arch)` is exported and documented as something one passes to `Optimizer` in
  place of `ZygotePullback`, so this is wrong training, silently, for every batch size above one.
  It is not new — `reduce = +` is what SymbolicNeuralNetworks 0.3 did too — and it is the same
  defect SymbolicNeuralNetworks records for `FeedForwardLoss` under *Open Issues → Semantics* in its
  own changelog, where it is called out as unfixable within a design that differentiates one
  symbolic sample.

  Closing it means choosing: make `HNNLoss` additive over the batch (it would no longer be scale
  invariant), or drop the symbolic pullback for architectures whose loss is not additive. Either is
  a decision about the loss, not a repair, which is why this release only documents it.

- **B6. Training a Hamiltonian neural network through `train!` is broken for every method.** All
  three call `vectorfield(nn, x, params)` — `ExactHnn` at `src/training_method/hnn_exact_method.jl:6`,
  `SEulerA` and `SEulerB` at `src/training_method/symplectic_euler.jl:13` and `:18` — and no such
  method exists. `vectorfield` resolves only to GeometricBase's methods on `AbstractStateVariable`
  and `State`, so the call raises a `MethodError` as soon as the first gradient is taken.

  Found by repairing `scripts/hnn_pendulum.jl`, which is several API generations behind and hid this
  behind four earlier failures (the two-argument `MomentumOptimizer`, the keyword
  `HamiltonianArchitecture` constructor, `∇H`/`dH` deleted from `scripts/pendulum.jl` while
  `get_data_set` still called them, and `get_data_set` returning a bare `(data, target)` pair where
  `train!` wants a `TrainingData`). Those four are fixed; the script still does not run to
  completion, and what stops it is this.

  Nothing under `test/` covers it: the HNN training methods are exercised only from `test/train!/`,
  which `runtests.jl` does not include. Closing this means deciding what `vectorfield` should be for
  a `NeuralNetwork{<:HamiltonianArchitecture}` — `hamiltonian_vector_field` is the obvious candidate
  — and giving it a test that runs.

  (There is no **B3** any more — it was `input_dimension`/`output_dimension` existing twice, closed
  by SymbolicNeuralNetworks 0.5; see *Fixed* above. The number is left vacant rather than reused.)

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

- **C7. `SymbolicPullback(::HamiltonianArchitecture)` duplicates the upstream constructor.** It has
  to, because `SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` derives the dimension of the
  loss's target from `output_dimension(nn.model)`, and for an HNN that is the scalar Hamiltonian
  rather than the vector field the loss compares against (see *Fixed*). Reproducing the constructor
  means reaching into three names that SymbolicNeuralNetworks does not export —
  `symbolic_parameter_gradient`, `ParameterGradient`, and the two-argument `SymbolicPullback` inner
  constructor — so an upstream refactor breaks GML silently at the type level. A keyword on the
  upstream constructor, or a `NetworkLoss` interface that states its own target dimension, would put
  this method back to one line.

- **C8. `scripts/` has been dead since SymbolicNeuralNetworks 0.2.** `scripts/test/test_symbolic.jl`
  calls `Symbolize` and `scripts/loss/build_loss.jl` calls `symbolic_params`; neither has existed
  for several breaking releases, and `scripts/Project.toml` is in no CI job, so nothing notices.
  Either port them or delete them — leaving them is the option that keeps costing a reader time.

- **C9. Seven include sites under `legacy/` name files that do not exist.** Six `legacy/hnn/`
  scripts include `../../scripts/data.jl` and `hnn_simple.jl` includes `../../src/training.jl`;
  neither file exists anywhere in the repository, and neither did before the move to `legacy/`. Of
  the 29 include targets under `legacy/`, the other 22 resolve. The spelling is now at least
  consistent with where the files sit, so what remains is a decision about `data.jl`: reconstruct it
  (it generated the pendulum training data, which `scripts/pendulum.jl` now does) or delete the
  scripts that need it.

### D. Unverified

Not defects — claims this release makes that nothing has actually checked yet.

- **D2. The test suite has never been run on Julia 1.10.** The package precompiles and loads there
  now, and resolves from the registry, but no suite has run there — and `julia = "1.10"` is a claim
  this release newly makes. The full runs so far were on 1.13.0-rc2. CI covers 1.10 and is the first
  thing that will exercise it.

- **D4. The upstream fix was measured on one optimizer.** The compile-time figures come from the
  `Adam` path. The quasi-Newton and Newton caches and states were widened on the strength of their
  *inferred types* — a sound argument, but not a measurement. Catalogued upstream as GeometricOptimizers C15.
