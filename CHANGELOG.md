# Changelog

All notable changes to GeometricMachineLearning.jl are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0, so a minor bump is a
breaking release).

> [!NOTE]
> Entries for 0.1.0 through 0.4.8 were reconstructed from git history, the release tags and the
> merged pull requests, not written at the time. They are accurate about *what* changed and are
> deliberately coarser about detail than the 0.5.0 section below, which was written
> alongside the work. Where a release removed exported names the list is given; where it is a
> reconstruction of intent, it says so.

## [Unreleased]

### Removed (breaking)

- **`RecurrentNeuralNetwork` and `LSTMNeuralNetwork`**, along with
  `src/architectures/recurrent_neural_network.jl`, `src/architectures/LSTM_neural_network.jl` and the
  driver scripts under `scripts/Script_using_fully_GML/{RNN,LSTM}/`.

  Both were already unusable. `Chain(::RecurrentNeuralNetwork)` returned an
  `AbstractNeuralNetworks.GridCell` rather than a `Chain`, and the cells it held still defined the
  pre-0.6 `initialparameters(cell, backend, T; init, rng)` signature, so building a `NeuralNetwork`
  from one errored out. Nothing in the test suite touched them. AbstractNeuralNetworks 0.7 removes
  `src/cells/` — `Recurrent`, `LSTM`, `GRU`, `IdentityCell` and `GridCell` — so the imports have to go
  regardless. The transformer-derived architectures are what GML maintains for time series; the docs
  had flagged the LSTM implementation as likely to be deprecated since it was written.

  `import AbstractNeuralNetworks: IdentityActivation, ZeroVector` went with them.
  `Chain(::RecurrentNeuralNetwork)` was the only thing in `src/` that named `IdentityActivation`,
  and `ZeroVector`'s last use moved to `legacy/` in `ec8e8fa5`. Neither was ever exported by GML,
  and `IdentityActivation` is an `AbstractNeuralNetworks` export that the blanket `using` at the top
  of the module already provides, so `GeometricMachineLearning.IdentityActivation` still resolves —
  the line was redundant, not load-bearing.

- **`NeuralNetworkParameters` is no longer exported; the name is `NetworkParameters`.** The parameter
  container moved out of `AbstractNeuralNetworks` into
  [NeuralNetworkParameters.jl](https://github.com/JuliaGNI/NeuralNetworkParameters.jl) in
  `AbstractNeuralNetworks` 0.7, which removed the old name outright rather than leaving an alias, so
  that one type has one name across the ecosystem. This package follows, at every call site — 31 in
  `src/`, `docs/` and `test/`, seven more in `scripts/` — and in its export list. It is the same type
  object, so `::Type{}` dispatch, `<:` bounds and `NetworkParameters{keys}(vals)` construction are
  unaffected; only the spelling changes.

  ```julia
  # before
  using GeometricMachineLearning        # brought NeuralNetworkParameters into scope
  # after
  using GeometricMachineLearning        # brings NetworkParameters into scope
  ```

- **The HDF5 extension no longer carries its own traversal.** Five `h5save` methods tagging a
  `gml_type` attribute, the `_gml_h5load` reader and the `_natural_sort_keys` key-order heuristic are
  gone — 73 of the extension's 187 lines. Each job now sits with the package that owns the pieces:

  - `NeuralNetworkParameters` walks the parameter set and writes it, recording each group's key order
    in a `keys` attribute. `_natural_sort_keys` was standing in for that, and it *guessed*: it sorted
    on a trailing integer when every name in the group had one and fell back to lexicographic order
    otherwise, so a group whose names do not end in a digit came back in whatever order sorting gave.
  - `GeometricOptimizers` says where each structured matrix keeps its numbers, through
    `freeparameters`/`rebuild`, and registers the types so a file loads with no prototype.
    `StiefelManifold` and `SymmetricMatrix` are its types, not this package's, so the methods here
    were type piracy twice over — on `h5save` and on the type.

  Existing files still load. `NeuralNetworkParameters` recognises the `gml_type` tag and rebuilds
  through the same registry, and `test/hdf5_support.jl` now writes a file in the old layout by hand
  and reads it back, so the deletion cannot quietly make old files unreadable.

### Fixed

- **Zygote 0.7 silently zeroed every gradient that flows through `assign_q_and_p`.** Its `rrule`
  built the cotangent of the split with `vcat(qp_diff...)`. Under Zygote 0.6 the two components had
  always been unthunked by the time they arrived, so that concatenated a `q` block and a `p` block
  into one gradient vector. Zygote 0.7 stopped unthunking eagerly, so the components arrive as
  `Thunk`s, and `vcat` of two thunks concatenates nothing — it builds a two-element `Vector{Thunk}`.
  Zygote then maps `unthunk` over *that*, and the caller gets a two-element `Vector{Vector{T}}`
  where a length-`2n` vector was due.

  `gradient` catches it (`ProjectTo` throws `DimensionMismatch`), but **`jacobian` does not**: it
  returns a matrix of zeros. Every symplectic-autoencoder and PSD path runs through
  `assign_q_and_p`, so the first thing the test suite reported on 0.7 was
  `test/psd_architecture_tests.jl` failing its symplecticity check against an all-zero Jacobian.

  The rule now unthunks the tangent and both of its components. On 0.7 the PSD decoder Jacobian is
  again bit-identical to what 0.6 produced.

- **Three kernel `rrule`s returned a cotangent of the wrong rank**, and the type piracy that
  covered for it is gone. `tensor_mat_mul`, `mat_tensor_mul` and `tensor_transpose_mat_mul` each
  take a *matrix* argument `B` and gave back an `n×m×1` array for it — `sum(_, dims = 3)` with the
  trailing singleton axis left on, where the sibling rules for the structured types (`lo_mat_mul`
  and friends) `reshape` it away. A cotangent has to have the shape of the primal it belongs to.

  That wrong rank is what forced the `ChainRules._adjoint_mat_pullback` method in
  `src/layers/multi_head_attention.jl`, whose own comment called it `# type pyracy!`: a 3-tensor
  method added to another package's unexported internal, so that differentiating
  `mat_tensor_mul(ps.PQ[key]', x)` in `MultiHeadAttention` would not hit a `MethodError`. Fixing
  the rank at the source makes it unnecessary.

  The pirated method was also, incidentally, the only reason the gradient of a *non*-manifold
  transformer weight came back as a `Matrix` rather than an `Adjoint{T, Matrix{T}}` — it
  materialised the transpose on the way through, where `ChainRules`' own rule does not. (This is not
  GML-specific: plain `Zygote.gradient(W -> sum(W' * x), W)` returns an `Adjoint` too.) Deleting it
  without more would have quietly changed the type of every such gradient and broken the invariant
  `test/transformer_related/transformer_gradient.jl` asserts, so the three rules now put their
  matrix cotangent through `_matrix_cotangent`, which fixes the rank *and* gives it the array type
  of the primal.

### Changed

- **The kernel `rrule`s honour the ChainRules interface for thunked cotangents.** Twelve pullbacks
  declared their cotangent as `::AbstractArray{T, 3}` — and the four `tensor_inverse` ones as `::AT`,
  the primal's *exact* array type, so even a plain `Array` cotangent for a `SubArray` primal was a
  `MethodError`. A `Thunk` satisfies none of those. They now take the tangent unconstrained and
  `unthunk` it.

  Eleven `f(::Thunk, ...)` forwarding methods existed to route around the same problem one call site
  at a time (`tensor_mat_mul(::Thunk, ::AbstractMatrix)`, `tensor_transpose(::Thunk)`,
  `augment_zeros(::Thunk, _)` and so on), each wrapping the kernel back up in a fresh `Thunk`. Two
  things were wrong with that. They dispatched on `Thunk` alone, so `InplaceableThunk` — which is an
  `AbstractThunk` but not a `Thunk`, and which ChainRules also emits — went straight past them into
  a `MethodError`. And they nested: the rule bodies already wrap the call in `@thunk`, so forwarding
  built a `Thunk` inside a `Thunk`, and `unthunk` removes one layer of thunking. The caller got a
  thunk where an array was due — the same silent failure as the `assign_q_and_p` case above.

  Unthunking where the tangent is consumed replaces all eleven, and handles both kinds of thunk.

  `test/custom_ad_rules/kernel_pullbacks.jl` had recorded the gap as six
  `check_thunked_output_tangent = false` opt-outs. Those are deleted, and the rules pass with the
  check on.

- `init_output`'s pullback had a `where T` on the inner function that shadowed the `T` of the `rrule`
  it sits in, on a signature it did not need — the body ignores its argument and returns
  `ZeroTangent()` regardless.

- **`save(filename, nn)` returns `filename`.** It used to return whatever the `h5open` block left
  behind — the value of the innermost `h5save`, an implementation detail of the traversal. Returning
  the path is what `NeuralNetworkParameters.save(filename, ps)` does, so the two now agree.

### Added

- **`load(NeuralNetwork, h5, arch, prototype)`** — a parameter set of the right shape to rebuild the
  structured leaves against. It is the form that needs no registration: `rebuild` has a prototype to
  take the non-differentiable fields from, so the file's type tags and
  `NeuralNetworkParameters.register_parameter_type!` are not consulted at all. Both the store and the
  filename overloads take it.

### Dependencies

- **`NeuralNetworkParameters = "0.1"`** added, and **`AbstractNeuralNetworks = "0.7"`** (was
  `"0.6.4"`). The parameter container is defined in the former as of the latter; see *Removed* above.

- **`GeometricOptimizers = "0.4.1"`** (was `"0.4"`). 0.4.1 is the release that carries the
  `NeuralNetworkParameters` leaf protocol for the manifolds, storage matrices and horizontal lifts,
  which is what lets this package's HDF5 extension drop its own copy of the traversal.

- **`SymbolicNeuralNetworks = "0.6"`** (was `"0.5"`). 0.5 caps `AbstractNeuralNetworks` at `"0.6.4 -
  0.6"`, so leaving the bound would have made this package's `[compat]` unsatisfiable rather than
  merely unresolved. 0.6 is the release that follows the container out to `NeuralNetworkParameters`.

  > **Merge order.** `AbstractNeuralNetworks` 0.7.0, `GeometricOptimizers` 0.4.1 and
  > `NeuralNetworkParameters` 0.1.1 are all in the General registry as of 2026-08-23.
  > `SymbolicNeuralNetworks` 0.6.0 is not: its `abstractneuralnetworks-0.7` branch still says
  > `0.5.0`, and still does `using AbstractNeuralNetworks: QPTOAT`, which 0.7 replaced with
  > `ArrayOrNamedTuple`, so it does not load as it stands. That is the one release this waits on.
  >
  > Until it lands, CI here fails at `Pkg.instantiate`. That is expected, not a regression.

- **`Zygote = "0.7"`** (was `"0.6"`). 0.7 replaced the eager unthunking in `wrap_chainrules_output`
  with `unthunk_tangent` at the `gradient`/`pullback` boundaries, which is what let thunks reach
  GML's `rrule`s and surfaced everything under *Fixed* above. Implicit parameters are deprecated in
  0.7; GML never used them, so nothing here changes on that account.

  0.6 is dropped rather than kept alongside. The resolver always takes the newest admissible
  version, so `"0.6, 0.7"` is a bound CI would never exercise — the same reasoning that dropped
  SymbolicNeuralNetworks 0.3 in 0.5.0. Zygote 0.7 requires Julia 1.10, which this package already
  does.

- **`ChainRules` dropped from `[deps]`.** Removing the `_adjoint_mat_pullback` piracy left it with
  no reference anywhere under `src/` or `ext/`; only `ChainRulesCore` is used, for `rrule`,
  `NoTangent`, `ZeroTangent`, `@thunk` and `unthunk`. It still arrives in the manifest through
  Zygote, so the rules it defines are loaded as before.

- **`LazyArrays` dropped from `[deps]`, and the exact pin with it.** Nothing under `src/`, `test/`,
  `ext/`, `docs/` or `scripts/` referenced it. The last use was `LazyArrays.Vcat` in
  `Base.vec(::StiefelLieAlgHorMatrix)`, which left with the Lie algebras when 0.5.0 stopped keeping a
  second copy of GeometricOptimizers' geometry, and `3d2b4887` had already turned the module-level
  `using` into a bare `import` ("Set LazyArrays to imported (not used)") — which is why 0.5.0's sweep
  for dead dependencies, done by grepping for `using`, kept this one.

  An exact pin on an unused package is not inert, either. GeometricOptimizers *does* use
  `LazyArrays.Vcat`, in `Base.vec(::AbstractLieAlgHorMatrix)`, and declares `LazyArrays = "2"`;
  versions resolve per environment, so GML's `"=2.3.2"` was holding the shared LazyArrays at its
  January 2025 release for the one package in the graph that has a use for it. It arrives through
  GeometricOptimizers now, free to float. This is also what
  [#187](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/187) asked about: CompatHelper
  proposed widening the entry to `"=2.3.2, 2"`, a range whose `=2.3.2` clause `2` already contains.

- **`ForwardDiff = "1"`** (was `"0.10, 1"`). GeometricOptimizers requires 1, so the resolver never
  had a reason to pick 0.10 and that branch was untestable — the same reasoning that dropped
  SymbolicNeuralNetworks 0.3 in 0.5.0. The dependency itself is real: `ForwardDiff.jacobian` in
  `src/reduced_system/reduced_system.jl` is its one call site.

  Together these close **C4**.

## [0.5.0] — 2026-08-19

**The optimizer machinery moves to [GeometricOptimizers][go].** GML no longer implements its own
optimizers: the methods, caches, states, global sections and retractions all come from
GeometricOptimizers, and GML keeps only the parts that are about neural networks — walking a
`NeuralNetworkParameters` tree, and the manifold layer types.

**Requires GeometricOptimizers 0.4.** The eleven geometry types GML used to define itself are
`import`ed from there now, and the interface it needs — `metric`, `check`, `Ω`, `global_section`,
`apply_section!`, `update_section!`, the retraction types, `AdamOptimizerWithDecay` — only became
public API in [GeometricOptimizers 0.4.0][go50]. GML does not load against 0.3.

This is a breaking release and the break is not mechanical. Read *Changed (breaking)* before
upgrading.

[go]: https://github.com/JuliaGNI/GeometricOptimizers.jl
[go45]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/45
[go50]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/50

### Removed (breaking)

- **GML's copies of eleven types GeometricOptimizers also defines.** `Manifold`, `StiefelManifold`,
  `GrassmannManifold`, `SkewSymMatrix`, `SymmetricMatrix`, `AbstractTriangular`, `LowerTriangular`,
  `UpperTriangular`, `AbstractLieAlgHorMatrix`, `StiefelLieAlgHorMatrix`,
  `GrassmannLieAlgHorMatrix` and `StiefelProjection` are now *imported* from GeometricOptimizers and
  re-exported. Twelve files go with them — all of `src/arrays/` bar `poisson_tensor.jl`, all of
  `src/manifolds/`, and `src/optimizers/go_bridges.jl` — about 2500 lines.

  The copies were near-verbatim, but Julia saw them as *distinct types*, so none of
  GeometricOptimizers' generic machinery dispatched on them: GML re-implemented
  `geodesic`, `cayley`, `apply_section!`, `global_rep` and `update_section!` once per manifold, and
  `go_bridges.jl` held some thirty more methods reconnecting the two hierarchies. All of that is
  gone. This closes **B2**
  ([#234](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/234)).

  `import` rather than `const X = GeometricOptimizers.X`: GML adds constructor methods to several of
  these types, and extending a type reached through `using` warns on every such method since Julia
  1.12.

  Not a source break for a caller — the names are still exported and mean the same thing — but the
  *types* are now GeometricOptimizers', so `x isa GeometricMachineLearning.StiefelManifold` and
  `x isa GeometricOptimizers.StiefelManifold` are the same question, where before they were
  different ones with different answers.

- **`AdamOptimizerWithDecay` is GeometricOptimizers'**, and GML's own is deleted. This closes **B1**:
  both packages exported the name, so `using GeometricMachineLearning, GeometricOptimizers` failed
  outright on it. It was the same algorithm — Adam's direction with a learning rate decaying by the
  same `γ = exp(log(η₂/η₁)/n)` — packaged differently, and upstream's packaging is the right one:
  the direction is an `Adam` method and the schedule is a `DecayingStatic` line search.

  **What a call has to change.** It is now a `(algorithm, linesearch)` pairing rather than an
  `OptimizerMethod`, so it splats into `Optimizer` instead of being passed positionally, `T` is
  positional and defaults to `Float64` rather than being taken from `η₁` (so `Float32`), and the
  moment coefficients are the keywords `β₁`, `β₂` rather than positional `ρ₁`, `ρ₂`:

  ```julia
  Optimizer(AdamOptimizerWithDecay(n_epochs), nn)                      # before
  Optimizer(nn; AdamOptimizerWithDecay(n_epochs, Float32)...)          # after
  ```

- **The optimizer caches stop being exported.** `AbstractCache`, `GradientCache`, `MomentumCache`
  and `AdamCache`. They are `solver_step!` scratch and stay internal to GeometricOptimizers, for
  every method alike; reach one as `GeometricOptimizers.AdamCache` if you genuinely need to name it.

- **`update!` stops being exported.** GML imported `AbstractNeuralNetworks.update!` and never added a
  method to it, so all the export did was shadow `GeometricOptimizers.update!` — a *different*
  generic function, and the one that actually has methods for the optimizer caches. That one is
  re-exported now instead.

- **`SymplecticLieAlgMatrix`, `SymplecticLieAlgHorMatrix` and `SymplecticProjection` stop being
  exported.** Nothing has defined them for as long as the git history goes back, so the exports were
  silent `UndefVarError`s waiting for a caller.

- **Twelve test files that duplicated GeometricOptimizers' suite**, under `test/arrays/`,
  `test/manifolds/` and `test/optimizers/utils/`. They tested the shared types, which upstream tests
  itself; what they covered and upstream did not was ported there first (see its changelog — it
  turned up four defects in the upstream suite, including a test file that never tested the Stiefel
  global section). `test/arrays/triangular.jl` keeps the half that tests GML's tensor kernels.

  Eight further files went with them — `test/optimizers/{exponential_retractions, riemannian_gradients,
  hor_lift, lie_alg_lifts, manifold_optim, momentum_optim_test, standard_optim_test}.jl` and
  `test/optimizers/manifold_related/legacy_functions.jl`. All were unreachable from `runtests.jl`,
  and most could not have run: two `include` paths deleted years ago, three `using Lux`.

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
  `_euclidean_update!` has no method for it and the step raises a `MethodError`. That work is not
  done.

  It did get cheaper, though. This entry used to add that bridging it needs `_fill!`,
  `_difference!`, `outer!` and the `ParameterHandling.flatten` round-trip *taught about GML's
  manifold and lift types*, because those were different types from GeometricOptimizers' and none of
  its `Manifold` methods applied. After the type unification above that half is simply gone —
  upstream's `flatten`, `_fill!`, `_difference!` and `outer!` already work on these types, because
  they are now the same types. What remains is routing `BFGS` through the per-leaf tree update at
  all, which is the same question as **C1**.

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

**The MNIST material moved to [GMLDatasets](https://github.com/JuliaGNI/GMLDatasets.jl).** That is
the last two entries above, along with the MNIST tutorial and the MNIST scripts. GML is a library
for scientific machine learning and had no business pulling an image-dataset package into its
documentation build to document itself — the MNIST tutorial *downloaded the data set every time the
docs were built*. `MLDatasets` is now absent from `docs/Project.toml` and `scripts/Project.toml`,
and the docs build is offline again.

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

What stayed: `ClassificationTransformer`, `ClassificationLayer`, `ClassificationTransformerLoss` and
`accuracy`. None of them is specific to image data, and they are what GMLDatasets' tutorial trains.

`BFGSOptimizer`, `BFGSCache`, `SymplecticStiefelManifold`, `default_optimizer`, `split_and_flatten`
and `onehotbatch` are the whole of the change to the exported surface, checked against
`names(GeometricMachineLearning)` rather than by reading the export list — the list spans
continuation lines, and reading it misses them.

### Changed

- **GeometricOptimizers 0.4.** The bound was `"0.2.1"` and the resolved version 0.2.2. It moved to
  `"0.3"` first and to `"0.4"` here, but 0.5.0 is the first release either lands in, so the only
  move a caller sees is `"0.2.1"` → `"0.4"`. Why the bound cannot stop at 0.3 is above: the
  interface this release imports only became public API in 0.4.0.

  GO 0.3.0 was a breaking release and **none of what it broke is reachable from here**. It renamed
  `_BFGS` and `_DFP` to `BFGS` and `DFP` and exported them together with `BFGSState`/`DFPState`, and
  it removed the exports `NewtonOptimizer`, `BFGSOptimizer` and `DFPOptimizer`, none of which had
  ever been defined. GML calls no name in either group — its quasi-Newton entry point was its own
  `BFGSOptimizer`, which this release removes for the reasons above — and `git diff v0.2.2..v0.3.1
  -- src/` in GO is that rename and its docstrings, and nothing else.

  This entry used to add that GO's `BFGS`/`DFP` exports could not collide because a blanket `using
  GeometricOptimizers` would make redefining the ~20 names GML defined itself an error on Julia
  1.10, so there was no blanket `using` to collide with. Neither half of that is true any more: the
  named `using` list is gone, and so are the types GML defined itself. The names still do not
  collide, for the plainer reason that GML neither imports nor exports either of them.

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
  1.9 was never satisfiable with this dependency set in practice. The claim is measured, not
  inferred: the full suite is green on 1.10 on Linux, macOS and Windows alike on the release tree
  ([CI run 32219315656](https://github.com/JuliaGNI/GeometricMachineLearning.jl/actions/runs/32219315656)),
  which is what closes what used to be open issue **D2**.

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

### Documentation

- **The `Manifolds` and `Optimizer` chapters move to GeometricOptimizers**, together with the two
  `Special Arrays and AD` pages whose data structures are its — `arrays/skew_symmetric_matrix.md` and
  `arrays/global_tangent_spaces.md`. Thirteen pages, ~3050 lines, documenting types that live there
  now. `arrays/tensors.md` and `pullbacks/computation_of_pullbacks.md` stay: they document GML's own
  tensor kernels and AD.

  `optimizers/optimizer_framework.md` splits. Its framework theory merges into upstream's
  `manifold_optimizers.md`; what is left is a new `optimizers/optimizer.md` covering GML's own
  `Optimizer`, `optimize_for_one_epoch!` and `optimization_step!` — the parameter tree and the
  training loop.

- **What the PDF book loses.** `_latex_pages` drops the whole `Background → Manifolds` chapter and
  the four-page `Optimizers` part, keeping a one-page `Optimizer` chapter, and the Appendix's
  `Special Arrays, Tensors and Pullbacks` becomes `Tensors and Pullbacks`. The book now opens on
  geometric structure and takes the manifold optimizers as given, citing them.

- **`DocumenterInterLinks`** enters `docs/Project.toml` and `docs/make.jl`, with a committed
  inventory under `docs/inventories/`. Thirty-six references from the chapters that stayed into the
  ones that moved are now real cross-references rather than dangling `@ref`s, and the seven
  de-referenced code spans C3 complained about (`𝔄`, `cayley`, `update!` …) can be links again.
  This closes **C3**.

### Fixed

- **`Matrix + SkewSymMatrix` was a `StackOverflowError`.** `Base.:+(B::AbstractMatrix,
  A::SkewSymMatrix)` read `B + A`, which is itself. Fixed by the type unification above: upstream's
  method, which reads `A + B`, has always been right. GeometricOptimizers' suite now asserts that
  addition against a dense matrix commutes, for all four structured types rather than for the one
  instance.

- **`parent(::StiefelLieAlgHorMatrix)` referenced an unbound variable.** It returned `(A, B)` where
  `B` was never defined — an `UndefVarError` for any caller. Also fixed by the unification;
  upstream returns `(A.A, A.B)`, which is what its `vec(::AbstractLieAlgHorMatrix)` builds on.

- **A decaying step size was read one step early.** `optimization_step!` read the step size *before*
  incrementing `opt.iterations`, so the first step of a run took `α(0) = η₁` where the pre-0.5
  `AdamOptimizerWithDecay` incremented first and took `α(1) = γη₁`. Every step of a run was therefore
  one place early in the schedule. The increment now comes first, which is also how
  `DecayingStatic` counts and how `GeometricOptimizers.solve!` counts (it calls
  `increase_iteration_number!` before `solver_step!`) — and what upstream's
  `test/adam_optimizer_with_decay.jl` asserts GML does. Pinned by `schedule_starts_at_one` in
  `test/optimizers/optimizer_convergence_tests/adam_with_learning_rate_decay.jl`.

  It affected only a decaying step size; a fixed one is the same at every `t`.

- **A pullback test asserted nothing.** The loop in `test/arrays/triangular.jl` comparing the batched
  `mat_tensor_mul` pullback against the single-slice one was written as bare expressions rather than
  `@test`s, so it ran and discarded its results. They are `@test`s now, and they pass.

- **`solve!` was a second generic function.** GML's `solve!(::NeuralNetwork{<:PSDArch}, …)` — solve
  for the parameters directly, by SVD, rather than training for them — created a new function of that
  name rather than adding a method to the one a caller already had. It is imported from
  GeometricOptimizers now, so `using GeometricMachineLearning, GeometricOptimizers` no longer
  collides on it either.

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

- **Four convergence tests seed per invocation instead of once per file.** `svd_optim.jl` and
  `sae_error_lower_than_psd_error.jl` failed on Julia 1.10 and 1.12 respectively, and neither was a
  numerical regression: given the same starting point the new optimizer stack agrees with the old to
  13 significant digits. Each file seeded once at the top and then called its helper *twice*, so the
  second call started from whatever RNG state the first happened to leave behind — and `Optimizer`
  now draws more randomness than it did. `GeometricOptimizers._similar` of a manifold parameter is
  `rand(manifold_constructor(a){T}, size(a)...)`, a fresh random point on the manifold, because
  upstream makes `Base.similar(::Manifold)` an error on the grounds that uninitialised storage is not
  a manifold point; `GradientState` allocates its `x̄` slot with it. GML's `StiefelManifold` *is*
  `GeometricOptimizers`' type now, so that method applies where on `main` the call fell through to
  `similar(a)` and GML's own `Base.similar(::StiefelManifold)`, which allocated uninitialised storage
  and drew nothing. Constructing one optimizer over two Stiefel weights draws six batches of normals
  where it drew four — the four `global_section` batches are unchanged, and each manifold parameter
  adds one random manifold point. Every draw after the first `Optimizer` construction shifted, and
  both tests were passing on a thin margin: the `svd_optim.jl` gradient run went from 2% above the
  optimum to 21%, against a 10% tolerance.

  Seeding each invocation makes the starting point independent of what ran before it. The two
  assertions clear by 23× and by 2.6–4.5%, and are now stable to 13 digits across 1.10, 1.12 and
  1.13 — 1.13 had been passing the autoencoder comparison by 0.7%, i.e. by luck. `psd_optim.jl` and
  `adam_with_learning_rate_decay.jl` have the same shape and get the same treatment; the latter's
  manifold run also goes from 32 to 128 epochs, because `AdamOptimizerWithDecay(n_epochs)` fixes
  `γ = exp(log(η₂/η₁)/n_epochs)` and a 32-epoch budget collapses the learning rate to `η₂` before
  the run has trained — the loss fell by under 2% on 1.13 and *rose* on 1.12. The unused
  `tol = .35` keyword of `sae_error_lower_than_psd_error.jl`'s `test_accuracy` is gone; the
  same-named helpers in `psd_architecture_tests.jl` and `symplectic_autoencoder_tests.jl` do use
  theirs and keep it.

- **`test/training_parameters.jl` seeds its second testset.** Its `step_size = 1e-2` half asserts
  `!all(loss_moving .== loss_moving[1])`, and the file's comment claimed the assertion needed no
  seed. That is true of the `step_size = 0` half and false of this one: `tra_ps_data` contains an
  all-zero trajectory, and a draw that takes only zero samples for all five runs gives a zero
  gradient every time and a loss array that never moves. Measured over 60 seeds it happens for one
  initialisation in sixty, on 1.10 and 1.12 alike — and it duly took out `Julia 1.12 - windows` on a
  commit that changed nothing but this file's neighbours in the CHANGELOG. Seeded at 123, where the
  loss spreads by 0.16 on all three versions.

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

  (**B1**, **B2**, **B3** and **B4** are all closed and their entries are gone: B1 and B2 by this
  release — the duplicated `AdamOptimizerWithDecay` and the split `Manifold`, both under *Removed
  (breaking)* — B3 by SymbolicNeuralNetworks 0.5, and B4 by `a427add1`, which repaired the
  documentation build. The numbers are left vacant rather than reused.)

### C. Follow-ups and cleanups

- **C1. The parameter-tree traversal still belongs upstream.** `_make_optimizer_cache`,
  `_make_optimizer_state`, `_tree_optim_step!`, `_leaf_optim_step!` and the bespoke
  `GMLEuclideanState` are GML implementations of what GeometricOptimizers supports natively for a
  single parameter. `GMLEuclideanState` in particular duplicates what `GradientState`,
  `MomentumState` and `AdamState` already do for a plain array.

  What has to go upstream is *not* a reuse of GeometricOptimizers' `Optimizer`: that one needs an
  `OptimizerProblem`, i.e. an objective function, and minibatch training has none — the gradient
  arrives from AD one batch at a time. It is a new entry point there, a
  gradient-supplied-externally step over a `NamedTuple` parameter tree. GML's `Optimizer` would then
  be the `NeuralNetwork` constructor and the training-loop functor, and nothing else.

  `Optimizer` is the one name still exported by both packages, so this is also what closes the last
  of B1's class of collision.

- **C2. Two `isa` branches remain in `_leaf_optim_step!`** (for `AdamState`/`MomentumState`).
  Measurement showed the traversal is not implicated in the compile-time problem, so this is tidying,
  and it disappears entirely if C1 lands first.

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

- **C10. Ten exported names are undefined.** `CPUDevice`, `Device`, `LinearSymplecticLayerP`,
  `LinearSymplecticLayerQ`, `ResidualLayer`, `aresame`, `convert_to_dev`, `description`, `symbol`
  and `timestep` are in an `export` list and defined nowhere, so
  `[n for n in names(GeometricMachineLearning) if !isdefined(GeometricMachineLearning, n)]` returns
  all ten. They are harmless in the sense that nothing breaks until someone reaches for one, at which
  point they get `UndefVarError` from a name the package advertises.

  This release removed the three that happened to sit in the export block it was already rewriting
  (`SymplecticLieAlgMatrix`, `SymplecticLieAlgHorMatrix`, `SymplecticProjection`), which is why the
  count is ten rather than thirteen. The rest are spread across the module and were left alone
  deliberately: each needs a decision — define it, or drop the export — and a few are load-bearing
  names in prose (`description` is `export`ed with the comment "from GeometricBase to print docs",
  and GeometricBase does define it, so that one is likely an `import` that was never written).

  `GeometricOptimizers`' `test/exports.jl` closes this whole class with one assertion over `names`;
  this package has no equivalent, and adding one is the actual fix.

- **C11. 41 test files are unreachable from `runtests.jl`.** By area: 20 under `performance_tests/`,
  5 `orthogonalization_procedures/`, 4 `train!/`, 2 `cuda/`, and 10 singletons (`training_phnn.jl`,
  `macro_testerror.jl`, `integrator/test_integrator.jl`, `attention_layer/`, `custom_ad_rules/`,
  `data/`, `kernels/`, `layers/`, `symplectic_autoencoders/`, `transformer_related/`).

  They are not all the same thing, which is why this is one issue and not a deletion. The
  `performance_tests/` and `cuda/` files need hardware the suite does not assume; the `train!/` files
  cover `train!`, which **B6** says is broken for every method, so they would fail if enabled; and
  the singletons are mostly stale. What they have in common is that nothing runs them, so nothing
  tells you when they rot — `test/optimizers/lie_alg_lifts.jl`, deleted in this release, had been
  including `../src/arrays/skew_sym.jl` since before that path stopped existing.

  This release deleted the eight that were `GeometricOptimizers` material *and* could not have run.
  The remainder needs a decision per group: register them behind an environment flag (the GPU and
  performance ones), fix the thing they test (`train!`), or delete them.

### D. Unverified

Not defects — claims this release makes that nothing has actually checked yet.

- **D4. The upstream fix was measured on one optimizer.** The compile-time figures come from the
  `Adam` path. The quasi-Newton and Newton caches and states were widened on the strength of their
  *inferred types* — a sound argument, but not a measurement. Catalogued upstream as GeometricOptimizers C15.
