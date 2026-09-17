# Changelog

All notable changes to GeometricMachineLearning.jl are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0, so a minor bump is a
breaking release).

> [!NOTE]
> Entries for 0.1.0 through 0.4.8 were reconstructed from git history, the release tags and the
> merged pull requests, not written at the time. They are accurate about *what* changed and are
> deliberately coarser about detail than the 0.5.0 and later sections below, which were written
> alongside the work. Where a release removed exported names the list is given; where it is a
> reconstruction of intent, it says so.

## [Unreleased] — 0.8.0

> [!NOTE]
> Not released. This was written as `[0.7.0]` before 0.6.1 and 0.7.0 were cut, and the number was
> taken by the release below; the work it describes -- `apply_toNT`, `_eltype`, `map_to_cpu` -- is
> still to do.

**The traversal of a parameter set now belongs to the package that owns the parameters, and the
traversal of a `NamedTuple` belongs to `Base`.** 0.6.0 handed the HDF5 walk over to
[NeuralNetworkParameters.jl][nnp] and `GeometricOptimizers`; this release finishes the job for the
remaining walks. `map_to_cpu` becomes one walk, `apply_toNT` turns out to have been `Base.map` all
along, and `_eltype` turns out to have been a hand-rolled `parameter_eltype`.

**It also makes the optimizer cache immune to a change coming in `GeometricOptimizers`**, which is
the half of this release with no visible effect today — see *Fixed*.

**Requires `GeometricOptimizers` 0.5**, which is where the structured types' `changebackend` and
`GlobalSection` methods now live — and, with it, `NeuralNetworkParameters` 0.2 and
`SymbolicNeuralNetworks` 0.7. The three floors move together because the environment does not
resolve otherwise: 0.5 drops the `ParameterHandling` shim and lets `NeuralNetworkParameters` do the
flattening, so the 0.2 container is required rather than preferred — it carries the element type of
its leaves as `NetworkParameters{T}`'s first type parameter, which is what lets a parameter set be
an `OptimizerSolution{T}` there — and `SymbolicNeuralNetworks` 0.6 permits only the 0.1 container,
so it cannot coexist with `GeometricOptimizers` 0.5.

### Removed (breaking)

- **`scripts/test/` is gone — 15 files, a third copy of the retired test suite.** Its own
  `runtests.jl` errored 9 testsets, and every file in it tested the `train!` subsystem this release
  deletes. Nothing replaces it: the package's suite is `test/`.

  This closes **C8**, whose remaining half was `scripts/test/test_symbolic.jl` calling `Symbolize`,
  a name no installed version of SymbolicNeuralNetworks defines. It is deleted rather than ported,
  because porting it would have meant rewriting it around three further retired names — it builds
  its input with `get_batch` from a `TrainingData` and names `SEulerA()` as its method — and what
  would survive that is the assertion that a `SymbolicNeuralNetwork` agrees with the network it
  wraps. `test/hamiltonian_neural_network_tests.jl` is the file that exercises the symbolic path
  today; it covers `SymbolicPullback` and not that agreement, so the coverage is not carried over.

- **`scripts/Script_using_fully_GML/hnn_script.jl` and `sympnet_script.jl` are gone**, each replaced
  by a tutorial that is built and checked with the documentation:
  `docs/src/tutorials/hamiltonian_neural_network.md` for the first,
  `docs/src/tutorials/sympnet_tutorial.md` for the second. `hnn_script.jl:30` read
  `hnn = NeuralNetwork(hnn, Float64)`, a self-reference that cannot ever have run.

- **The `train!` subsystem is gone.** `src/training/` (7 files), `src/nnsolution/` (3), `src/data/`
  (5), `src/training_method/` (7) and `src/architectures/default_architecture.jl` are deleted —
  1,438 lines of `src/` — together with their `include` sites and export blocks. The package now
  exports **130 names where it exported 218**: 88 removed, none added, counted from
  `names(GeometricMachineLearning)` before and after rather than from the diff, which over-reports
  because some names are exported on more than one line.

  `DataLoader` + `Batch` + `Optimizer` is the training path. What the harness could do that it
  cannot is nothing, now that `LNNLoss`, `SymplecticEulerLoss` and
  `VariationalMidpointLoss` exist — those three ports are the reason this deletion is not a loss of
  capability, and they landed first for exactly that reason.

  Gone with it: `train!`, `TrainingData`, `TrainingMethod`, `TrainingParameters`, `TrainingSet`,
  `NeuralNetSolution`, `EnsembleNeuralNetSolution`, `EnsembleTraining`, `History`, `SingleHistory`,
  the data shapes and symbols (`TrajectoryData`, `SampledData`, `PositionSymbol`,
  `PhaseSpaceSymbol`, `DerivativePhaseSpaceSymbol`, `PosVeloSymbol`, `PosVeloAccSymbol`,
  `DataSymbol`), the six training methods and their constructors (`SEuler`, `SEulerA`, `SEulerB`,
  `ExactHnn`, `ExactLnn`, `VariaMidPoint`, `BasicSympNet`), `default_arch`, `default_method`,
  `matching`, `loss_single`, `loss_gradient`, and the accessors that went with them.

  **Note that `src/data/batch.jl` was the old batch machinery.** The modern `Batch` is
  `src/data_loader/batch.jl` and is untouched.

  **`Distances` is no longer a dependency.** `sqeuclidean` was the default distance of every
  `TrainingMethod`, and nothing under `src/`, `test/`, `docs/src/` or `ext/` reaches for the package
  once those are gone. Both the `[deps]` and the `[compat]` entry go with the `using`.

  **`∇q∇q̇L` is gone from `src/architectures/lagrangian_neural_network.jl`** as well. It is
  unexported, and its one caller was `src/training_method/lnn_exact_method.jl:10`. What it returned
  is the mixed Hessian block `LNNLoss` now takes from a compiled symbolic expression instead of from
  `Zygote`, so keeping it would leave two implementations of one formula. Its three neighbours in
  that file — `∇L`, `∇∇L` and `∇q̇∇q̇L`, plus the constant `DEFAULT_LNN_NRUNS` — were dead before
  this release and are left alone; see *C13* under *Open Issues*.

- **Seven exported names that were defined nowhere are no longer exported**, and one that should
  have resolved now does. `Device`, `CPUDevice`, `convert_to_dev`, `ResidualLayer`,
  `LinearSymplecticLayerP` and `LinearSymplecticLayerQ` had no definition anywhere under `src/` —
  reaching for any of them raised `UndefVarError` from a name the package advertised. `timestep`
  went with the block that exported it. `description` keeps its export and gains the
  `import GeometricBase: description` it always needed: `GeometricBase` defines that generic but does
  not export it, so `using GeometricBase` never brought it into scope.

  This closes **C10**. `test/exports.jl`'s allowlist is now **empty**, so the guard asserts outright
  that every exported name resolves, with no exceptions to trust. The one name here that has a
  definition is `ResidualLayer`, at `legacy/layers/resnet.jl`, which nothing under `src/` includes;
  the loaded layer of that shape is `ResNetLayer`.

- **The `train!` tests are gone, and six of them were passing.** Deleted: `test/train!/` (5 files),
  `test/integrator/test_integrator.jl`, `test/training_parameters.jl`, `test/data/test_data.jl`,
  `test/data/test_batch.jl`, `test/data/test_matching.jl`, `test/data/data_generation.jl` and
  `test/macro_testerror.jl` — 820 lines.

  **What that removes is real coverage, not just dead files.** `test/data/test_data.jl`,
  `test/data/test_batch.jl`, `test/data/test_matching.jl` and `test/training_parameters.jl` ran on
  every suite and passed: they covered `TrainingData` construction from arrays and from a
  `GeometricSolution`, the old `get_batch` partitioning, `matching` between a network and a data
  set, and `TrainingParameters` including the step size handed to `train!`. `test/macro_testerror.jl`
  covered a test-only macro. All of it tested code that no longer exists, so none of it could be
  kept — but the subsystem leaves with its tests, rather than its tests having been absent.

  The five files under `test/train!/` and `test/integrator/test_integrator.jl` were **not** running:
  `runtests.jl` never included them, and *C11* recorded them as unreachable. `test/reachability.jl`'s
  allowlist is now **empty** as well, so every file under `test/` is reached from `runtests.jl`.

- **The five `changebackend` methods for `GeometricOptimizers`' types are gone, and so is
  `GeometricOptimizers.GlobalSection(::NetworkParameters)`.** Both were type piracy of the same shape:
  the generic belongs to one package, the types to another, and this package owns neither. Both now
  live in `GeometricOptimizers` 0.5, which is what the compat floor moves for.

  The `changebackend` methods also sat inside the **HDF5 extension**, which had nothing to do with
  HDF5 — so `changebackend(GPU(), nn)` on a network with a manifold weight was a `MethodError` unless
  HDF5 happened to be loaded. And they enumerated five types where upstream dispatches on
  `Manifold`, `VectorStorageMatrix` and `AbstractLieAlgHorMatrix` in one method, so three families
  had no method here at all: `GrassmannManifold` and both horizontal lifts.

  Nothing under `src/` referenced `changebackend`, and the tests needed no rewriting — they import it
  from `AbstractNeuralNetworks`, so upstream's methods answer by dispatch. They did move, to
  `test/changebackend_tests.jl`, because coverage of a device transfer has no business being
  reachable only through the HDF5 testset; the three previously uncovered families are asserted
  there now.

- **`apply_toNT` is gone from the export list and from the package. It was `Base.map`.**

  ```julia
  apply_toNT(f, a, b)   →   map(f, a, b)
  ```

  `map` over `NamedTuple`s takes any number of arguments and already throws
  `ArgumentError: Named tuple names do not match.` on mismatched *or* reordered keys — which is what
  the hand-rolled `@assert keys(ps[1]) == keys(p)` was approximating, except that Base's check cannot
  be compiled out the way an `@assert` can. Heterogeneous values map fine, so a `StiefelManifold`
  beside an ordinary `Matrix` is no obstacle. Verified on Julia 1.10, the compat floor, as well as on
  1.13.

  `_norm`, `_diff` and `_add` use `map` directly; that is the faithful translation, not a
  simplification, because `_diff` and `_add` recurse through their own `NamedTuple` methods and
  `_norm` divides by `√length` one level down. `GeometricOptimizers` carried a
  character-identical copy of the same function, reached from here by qualified call; that copy goes
  in its own release, and this change is what frees it.

- **`_eltype` is gone; `NeuralNetworkParameters.parameter_eltype` replaces it.** The two are not the
  same function — `_eltype` returned the element type of the *first* leaf and read a structured leaf's
  dense interface, where `parameter_eltype` promotes across every leaf and descends through
  `freeparameters` — but at the four call sites this package had they cannot disagree, and it is worth
  saying why rather than claiming a fix that could not fire.

  `_eltype` was only ever asked for a `T` in two places: under `_use_go_cache`, which requires
  `x isa GeometricOptimizers.OptimizerSolution`, and on the `ps_leaf` that reaches
  `_leaf_optim_step!`, which is such an `x`. And `OptimizerSolution{T}` is homogeneous in `T` by
  construction — its `NamedTuple` arm is
  `ArrayNamedTuple{T} = NamedTuple{S,<:Tuple{Vararg{AbstractArray{T}}}}`. A layer mixing `Float32`
  and `Float64` weights therefore *fails* that test and recurses to one cache per weight, so a
  first-leaf answer and a promoted one were never different answers. What the substitution buys is
  four fewer methods to own and the upstream spelling at the point where the walk is upstream's.
  Unexported, so this is breaking only for code reaching into the package.

- **`add!(::NamedTuple, ::NamedTuple, ::NamedTuple)` is gone, and `_add` and `add!` are gone from the
  export list.** The container arm of `add!` had no caller in the package, the tests, the docs or the
  scripts, and `AbstractNeuralNetworks.add!` — whose generic it was a method of — is about a
  destination and two summands, which a parameter *tree* is not. `add!` remains available from the
  package that owns the generic, this one only adding methods for the structured matrix types:

  ```julia
  using AbstractNeuralNetworks: add!
  ```

  `_add`'s two siblings `_norm` and `_diff` were never exported, and they are the two of the three
  that anything in `src/` actually calls; `_add` was the odd one out. Qualified, it still works.

- **`_add(::History, ::SingleHistory)` is now `_push_history!`.** Unexported and internal, one caller.
  Two unrelated meanings on one name is one too many, and the new name says that it mutates its first
  argument, which the old one hid.

### Changed

- **The pendulum scripts train through `DataLoader` + `Batch` + `Optimizer` now, and they run.**
  `scripts/reproduction/hnn_pendulum.jl` is where *B6* was diagnosed, and this file recorded that it
  "still does not run to completion, and what stops it is this". It now runs to completion on
  `HNNLoss`: the training loss falls from `1.0017` to `0.0041` over 2000 epochs and the script
  writes its plot.

  `scripts/reproduction/lnn_pendulum.jl` is rebuilt on `LNNLoss`. It had never run under any API
  generation — it referenced `TrainingIntegrator` and `DataTrajectory`, neither of which exists,
  and included `../data_problem.jl`, a path that does not exist either. It now trains a Lagrangian
  neural network on the pendulum, `3.248` down to `0.175` over 200 epochs.

  `LNNLoss` needs a larger step size than `HNNLoss` — `1e-2` against `1e-3` — because its gradient
  reaches the parameters through a solve against the velocity Hessian, where `HNNLoss` is linear in
  the gradient of the learned Hamiltonian. At `1e-3` the loss moved from `1.190` to `1.188` over 3%
  of a run whose own estimate was 48 minutes.

  `get_data_set` in `scripts/utilities/pendulum.jl` returns `(input, output)` matrices for
  `DataLoader(input, output)` instead of a `TrainingData`. `pendulum_data`, which `README.md:29`
  depends on, is unchanged.

- **`get_LNN_data` in `scripts/utilities/data_problem.jl` was missing a transpose.** It computes the
  Euler–Lagrange acceleration as `inv(∇q̇∇q̇L) * (∇qL - ∇q∇q̇L * q̇)`, but `∇q∇q̇L` is
  indexed `[i, j] = ∂²L/∂qᵢ∂q̇ⱼ` and the chain rule contracts the *first* index, so the term is
  `∇q∇q̇L' * q̇`. The two agree exactly for the pendulum, whose position and velocity do not couple,
  and that is the only Lagrangian in `dict_problem_L` — so nothing this function ever produced was
  wrong. It is corrected because `LNNLoss` solves the same equation in `src/`, and one correct copy
  of a formula beside one wrong copy is worse than either alone.

- **`map_to_cpu` is one walk instead of eight methods.** `NeuralNetworkParameters.mapstorage` hands a
  function the storage of a leaf and rebuilds the leaf around the result, so the five methods that
  existed to unwrap and reconstruct a `StiefelManifold`, a `SymmetricMatrix`, a `SkewSymMatrix` and
  the two triangular types collapse into one, the two that recursed into a `NetworkParameters` and a
  layer go with them, and the plain-array one is all that is left — as `_to_host`, the function handed
  to the walk. `GeometricOptimizers` supplies the protocol for its own types, so nothing here knows
  which structured types exist and one added upstream is covered without a change on this side.

  `mapstorage` and not `mapparameters`: the latter hands the function *whole* leaves, which would
  still need a method per type to reach the storage. The `NeuralNetwork` method stays — the docs
  tutorials and several scripts call it on a whole network.

  It was **untested**, which is why the rewrite comes with `test/map_to_cpu_tests.jl`: that every
  structured leaf comes back as the type it went in as, that the `n` a structured leaf carries
  survives although it is not in its storage, that element types are preserved, that the leaves are
  copies rather than the same arrays, and that a whole network keeps its architecture, model and
  backend.

- **Every tracked file is Unicode NFC-normalised.** Nineteen files across `src/`, `test/`,
  `scripts/` and `docs/` stored `Ñ` (39 times), `ṗ` (18), `Ũ` (10) and `ȳ` (3) as a base letter plus
  a combining mark, inherited from macOS rather than chosen. `q̇`, `q̈`, `f̄` and `b̄` have no
  precomposed codepoint and are unchanged.

  Nothing about the compiled code changes: Julia's parser normalises identifiers to NFC, so the
  symbols were already precomposed and dispatch, field names and method resolution are untouched.
  No string literal was affected, and no changed line falls inside a `jldoctest` block. Two changed
  lines do sit in fenced blocks — `docs/src/tutorials/symplectic_autoencoder.md:65` in an
  `@example` block, and `docs/src/reduced_order_modeling/reduced_order_modeling.md:79` in an
  `@eval` block, which Documenter does execute. Both are accesses to `GeometricProblems`' `Ñ`,
  which is defined precomposed there, so both blocks run exactly as before; Documenter compares the
  output of neither.

  What changes is that the source now matches what a keyboard, an editor search or a `grep` pattern
  produces; in an NFD file a pattern typed in NFC matches nothing at all, silently. Every changed
  file is exactly the NFC normalisation of its predecessor.

- **The four `tensor_inverseN` kernels are regenerated with common-subexpression elimination.**
  `Symbolics.build_function` defaults `cse` to `false`, and without it each entry of the inverse is
  one deeply nested expression. JuliaFormatter then indents every level of that nesting — up to 512
  columns on the 5×5 — at a cost quadratic in the depth. The 5×5 kernel was 25 495 007 bytes and
  91 625 lines, of which **97.8 % was leading whitespace**; with CSE it is 26 657 bytes and 487
  lines. All four generated kernels together now come to 42 510 bytes, down from 25 586 095.

  **This unblocks the commit hook.** `fatou lint` did not terminate on
  `src/kernels/inverses/inverse_5x5.jl`, nor on `src/GeometricMachineLearning.jl`, which reaches it
  through the include chain. The shared `.githooks/pre-commit` runs `fatou lint` on staged files
  with no timeout, so staging either file hung the commit. The regenerated 5×5 lints in 45 ms, and
  the whole `src/` tree in 0.1 s. Every checkout, working tree and `Pkg.add` also stops carrying a
  25 MB file. The clone barely moves, and that is the answer to whether the old revisions are worth
  rewriting out of `.git`: they are not. The 25 495 007-byte blob packs to 464 650 bytes, and all
  five revisions of the file together come to 0.5 MB of a 420 MB pack.

  **The 5×5 is also about thirty times faster.** Without CSE the kernel rebuilds the whole
  determinant for each of the 25 entries: one output entry alone is 624 multiplications, the fully
  expanded 120-monomial Leibniz sum. With CSE the determinant is one shared temporary and the whole
  kernel is 419 multiplications. `tensor_inverse5!` drops from 969.7 ns to 31.7 ns per slice, and
  its first call in a fresh process from 10.785 s to 0.350 s — ten seconds of compile time, for one
  kernel. The 4×4 goes from 55.8 ns to 11.1 ns and the 3×3 from 12.2 ns to 3.1 ns. `Float64`, 4096
  slices, minimum of 200 repeats after warm-up, a cold process each time.

  The expressions are different, so results move in the last digits, but the inverse is the same
  one. Checked against `LinearAlgebra.inv` for all four sizes in `Float64` and `Float32`: maximum
  relative error 7.1e-15 and 2.5e-6 over 64 random well-conditioned slices per size, and 8.0e-13
  over 2000 `rand(5, 5)` slices. `tensor_cayley5` output is orthogonal to 1.5e-15 with a determinant
  of 1. Against a `Float64` reference over 2000 slices per size, taken over the whole draw, mean
  relative error is the same or lower than before at every size and precision: in `Float16`, 5.6e-4
  against 6.0e-4 (3×3), 9.1e-4 against 9.3e-4 (4×4) and 1.9e-3 against 3.0e-3 (5×5); in `Float32`,
  2.1e-7 against 4.6e-7 (5×5). The 2×2 is bit-identical.

  **Where it is worse is the ill-conditioned tail, and only there.** Over 4000 slices per size with
  condition number above 100, three cases regress slightly: `Float64` 3×3 (2.4e-14 against 1.7e-14),
  `Float32` 3×3 (7.9e-6 against 6.5e-6) and `Float32` 4×4 (4.6e-6 against 4.1e-6). In `Float16` the
  5×5 overflows to `Inf` on two slices in 4000, at condition numbers 4.5e3 and 7.8e3. That is not a
  zero denominator — the determinant is a normal number there — but inverse entries of order 1e4
  against a `Float16` ceiling of 65504. The failure is not new in kind: the old kernel overflows the
  same way at 3×3, which this change fixes, and at 2×2 the two overflow on the same slice, that size
  being bit-identical. None of it is reachable from the package. `tensor_inverse5` is called only
  from `tensor_cayley5`, whose `I + S` has a condition number between 1.08 and 2.18.

- **`volume_preserving_attention_tests` compares each determinant against the exact one instead of
  against the other.** It asserted `det₁ ≈ det₂` and then `det₂ ≈ det₃` — two independently computed
  approximations, so their errors add. At N = 3 in `Float16` each is about 2 % out, inside the
  3.1 % `Float16` tolerance on its own; the second assertion passed only while the two errors
  pointed the same way. It now asserts `det₁ ≈ det₂` and `det₁ ≈ det₃`, which is what
  volume preservation means and which does not double the error budget. The 2 % is not new: `det₃`'s
  deviation is bit-identical before and after the kernel regeneration above.

  **The 5×5 had no test coverage at all.** `test55_inverse()` and `test55_inverse_pullback()` in
  `test/kernels/tensor_inverse.jl`, and `test_tensor_cayley5` in `test/kernels/tensor_cayley.jl`,
  were defined and never called. All three are enabled and pass.

  **The `invNN_kernel!` kernels no longer constrain their two arguments to the same array type.**
  The 2×2, 3×3 and 4×4 were `(ˍ₋out::AT, A::AT) where {T, AT <: AbstractArray{T, 3}}`, so
  `tensor_inverseN!(out, A)` with a plain `Array` output and a `SubArray` input was a `MethodError`
  — the same defect class as the `::AT` cotangent signatures above. They now match the 5×5, which
  never carried the annotation.

  **The generator now sits beside what it generates**, as
  `src/kernels/inverses/inverse_generator.jl` rather than `legacy/codegen/matrixinverse.jl`. It is
  not legacy and it is not dead: it is the authoritative source of the four committed kernels, and
  anyone editing one of them needs to find it. Nothing includes it, and `Symbolics` stays out of the
  package's dependencies — the kernels are committed precisely so that building the package never
  needs a symbolic stack.

  It emits the complete kernel file — the slice index, the Cartesian output index, the
  `tensor_inverseN` wrappers and the `rrule` — rather than a `build_function` body that a human then
  wraps, so regenerating it reproduces what is committed.

- **`scripts/loss/` is deleted, and with it the symbolic training-loss experiment.** The four files
  were 3,391 lines, 39% of `scripts/`. What is abandoned is an attempt to differentiate a `train!`
  training-method loss *symbolically* and commit the result. `build_loss.jl` holds a
  `build_gradloss` that builds `Symbolics.gradient(los(ti, nn, …))` and returns the pair
  `(sloss, ∇loss)`. `build_loss_test.jl` carries its own copy of `build_gradloss`, and it is that
  copy which renames the emitted function to `∇loss_single` and writes the method source to a
  hard-coded absolute path under a previous author's home directory, after comparing the raw
  `build_function` output against the rewritten form of the same symbolic expression (`r1==r2`).
  `write_loss_1.jl` (2,909 lines) and `write_loss_2.jl` (298) are two such emissions.

  Two facts decided the deletion. `build_loss.jl` calls `symbolic_params`, which no installed
  version of SymbolicNeuralNetworks defines, so the generator does not run here. Both emissions
  dispatch on
  `TrainingIntegrator{SymplecticEulerA, PhaseSpaceSymbol, TrajectoryData, SqEuclidean}`, and no file
  under `src/` defines `TrainingIntegrator`, so they do not load. Whether either half could be
  repaired was not investigated, and this entry claims nothing either way.

  The emissions were also not general: each is frozen onto one concrete signature, a
  `NeuralNetwork{HamiltonianArchitecture{typeof(tanh)}, Chain{…}}` of fixed width and depth, so a
  network of any other shape had no method. This is the opposite case to `inverse_generator.jl`
  above, which is kept because it still reproduces what is committed beside it.

  Symbolic differentiation of a loss is not lost as a capability — SymbolicNeuralNetworks'
  `SymbolicPullback`, which this package already uses, is where it lives now. What is lost is this
  particular ahead-of-time, commit-the-gradient approach to it.

  **A sweep for orphans found three**, none repaired here. It covered the names the deleted files
  defined, the packages they imported, the names they consumed from `src/`, and references to the
  deleted paths and filenames — of which it found none. It is not a guarantee that nothing else was
  missed.

  `src/training/train.jl:13-14` holds a commented-out `loss_gradient` method — the signature on `:13`
  and the body on `:14` — whose body calls `∇loss_single`, a name the two emitted files defined. The
  lines are left exactly as they were, inert. The `loss_single` generic in `src/training_method/` is
  a *different* function, still defined, still called there and still exported.

  `src/utils.jl:19-27` defines eight unexported methods — four of `rdevelop`, four of `develop` —
  whose only references outside their own definition lines were in `build_loss.jl` (`:11`, `:13`,
  `:14`, `:20`, `:25`). The `develop` in `scripts/test/test.jl` is that script's own local
  definition, `build_loss_test.jl:5` took its `develop` from its `using` list rather than from
  `src/utils.jl`, and the `Pkg.develop` calls elsewhere in the repository are a different function.
  They are left in place.

  `Distances` in `scripts/Project.toml` is left declared but is no longer used under `scripts/`. The
  references the sweep found were `using Distances` at `build_loss_test.jl:8`, two `sqeuclidean`
  calls at `build_loss_test.jl:101`, and the `SqEuclidean` type parameter in the emitted files.
  Nothing under `scripts/` uses `BenchmarkTools` or `KernelAbstractions` either, but that was already
  so before this change — `Distances` is the one this deletion newly orphans.

- **Six research one-off scripts are deleted.** Each named a symbol the package no longer has, or a
  data file that does not exist anywhere in this tree; running each one confirmed the failure.

  - `scripts/ForcedHamiltonianSystem.jl` and `scripts/TimeDependentHarmonicOscillator_Analytic.jl`
    import `QPT2` from `GeometricMachineLearning`, and the second also imports `ParametricLoss`;
    neither name is defined or exported anywhere in `src/`. The first also calls `ForcedSympNet`,
    likewise absent. Both raise `UndefVarError: QPT2 not defined`, not at the `using` line but at
    the docstring'd function definition that uses `QPT2` as a type parameter (line 37 of the first
    script, line 49 of the second).
  - `scripts/normal_forms/non_rev_ham.jl`'s first failure is `Chain` at line 4, a three-way
    ambiguity: `Lux.Chain`, `AbstractNeuralNetworks.Chain` (re-exported by
    `GeometricMachineLearning`), and `SymbolicUtils.Rewriters.Chain`, declared `public` and loaded
    transitively — the script never `using`s `SymbolicUtils` itself. Patching that around shows the
    next blocker is
    `Gradient`, called bare but never exported (`GradientLayerQ`/`GradientLayerP` are the exported
    names). `SymplecticMatrix`, removed from the package in `444e6fac` (2023-05-31), sits only
    inside a function the script never calls, so it is not what stops this script.
  - `scripts/psd_auto_toda.jl` calls `SymplecticMatrix`, `SymplecticStiefelLayer` (never exported —
    `src/GeometricMachineLearning.jl:189` says so directly) and `StandardOptimizer`, none of which
    exist in `src/`. Even the 2026-08-16 commit that replaced this file's `GLMakie`/`Plots` calls
    with `CairoMakie` left these breaks in place.
  - `scripts/particles.jl` and `scripts/particles_cuda.jl` read
    `../[../]ReducedBasisMethods/runs/BoT_Np5e4_k_010_050_np_10_T25.h5`. The sibling
    `Packages/ReducedBasisMethods` repository exists but has no `runs/` directory and no file of
    that name in its git history; both scripts fail at `h5open` before either reaches a training
    step, and `particles_cuda.jl` fails there before any CUDA call.

  None of the six has a replacement inside the package a reader could substitute, so none is a
  one-line repair. This is filed here rather than under *Removed (breaking)*: `scripts/` is not part
  of the package, so no user of `GeometricMachineLearning` can break on it — the same reasoning that
  put the `scripts/loss/` deletion above in this section.

  `NLsolve` and `NNlib` in `scripts/Project.toml` are left declared but are no longer used under
  `scripts/`, exactly as `Distances` is above. The references were `using NLsolve` at
  `psd_auto_toda.jl:5`, and `using NNlib: relu` at `ForcedHamiltonianSystem.jl:5` and
  `TimeDependentHarmonicOscillator_Analytic.jl:6`. A grep for either name over the surviving
  `scripts/` tree now returns only the `[deps]` lines themselves.

- **`scripts/reproduction/harmonic_oscillator.jl` trains through `DataLoader` + `Batch` +
  `Optimizer` instead of `TrainingData` + `TrainingSet` + `train!`.** The old path no longer ran: it
  failed inside `src/data/data_training.jl:57`, `UndefVarError: timestep not defined in
  GeometricMachineLearning`, before ever reaching the neural network. `BasicSympNetMethod` needed no
  modern successor — `GSympNet` already trains through the generic `Optimizer` functor, as
  `scripts/reproduction/sympnets/sympnet_toda_lattice.jl` already does — so this is the same
  substitution, not a new one. The script now builds `DataLoader(ensemble_solution)` directly (a
  method for exactly this `EnsembleSolution` shape already exists at
  `src/data_loader/data_loader.jl:367`).

  **`plots.jl`'s seven plotting functions are retyped onto `DataLoader` and `NeuralNetwork`, and
  `plot_result` is called again.** Each keeps its original purpose — the two-form `plot_*!`/`plot_*`
  convention is unchanged — and only how it reaches its data moves: `_trajectory` reads a
  trajectory straight out of `dl.input.q`/`dl.input.p` instead of `get_data`;
  `plot_verification!` and `plot_prediction!` roll a trajectory out with `iterate(nn, ...)` instead
  of repeated calls to `nns.nn`; `plot_result`'s bounding box and trajectory sampling read
  `dl.n_params` and `dl.input` instead of `get_nb_trajectory`/`get_data`. The script now ends by
  calling `plot_result(dl, nn, H; ...)` again, producing the same four-panel PNG
  (`GSympNet_4-10_on_Harmonic_Oscillator.png`) as before, verified by running it and inspecting the
  four panels. The whole script runs to completion in about 40 s in an isolated process.

  Two details of `plot_result` are not a straight translation. The bounding box is now
  `minimum(dl.input.q[:, 1, :])` and its three siblings, over **every** component of every initial
  condition. The old code built a vector of `NamedTuple`s and called `min(vectors...)`, which
  compares vectors lexicographically — for the one-degree-of-freedom harmonic oscillator this is
  the same number, but the lexicographic form is not what a bounding box wants. And the first
  panel's title is `"Data"`, not `"Datas"`.

  **`plot_loss` is implemented rather than left an empty stub**, and `plot_prediction!` no longer
  takes a `DataLoader`. `plot_loss!`/`plot_loss` follow the file's two-form convention and draw the
  `loss_array` the `Optimizer` functor returns on a logarithmic axis; the script saves it as
  `GSympNet_4-10_on_Harmonic_Oscillator_loss.png`, which is what
  `scripts/reproduction/sympnets/sympnet_toda_lattice.jl` already does with its own loss array.
  Before this, `loss_array` was bound and never read. `plot_prediction!` and `plot_prediction` drop
  their `dl` argument because neither body ever touched it — every initial condition reaches them
  through `initial_cond`.

  `scripts/reproduction/harmonic_oscillator.jl` also drops `using GeometricSolutions` and
  `using GeometricEquations`. No name from either package appears in the file: `hodeensemble` and
  `exact_solution` both come from `GeometricProblems.HarmonicOscillator`.

### Fixed

- **`ReducedLoss` is trainable through the `Optimizer` functor.** Its functor annotated the
  parameter argument `params::NetworkParameters`, and it was the only loss in
  `src/loss/losses.jl` that annotated it at all. `Zygote.pullback` evaluates the forward pass with
  the parameters **unwrapped** — the closure body receives the underlying `NamedTuple`, not the
  `NetworkParameters` handed to `pullback` — so the annotated method did not match there and the
  call fell through to the untyped `NetworkLoss` fallback in `AbstractNeuralNetworks`. That
  fallback's body is an `error`, so what a caller saw was `Functor not defined for NetworkLoss of
  type ReducedLoss{…}` rather than a `MethodError` naming the argument types.

  Measured, with one closure and one argument: called plainly it receives a `NetworkParameters`,
  and through `Zygote.pullback` it receives a `NamedTuple`. The failure is therefore in the
  forward pass, which is also what the original stack trace shows — Zygote differentiating the
  `error` call, having already selected the fallback going forwards. The reverse pass is not
  implicated: the tangent comes back as a `NetworkParameters`, correctly wrapped.

  This is the path the symplectic autoencoder tutorial documents, so it was wrong for every user
  who followed it, not only for the two scripts here. Nothing had caught it because nothing ran
  it: the only `ReducedLoss` use under `test/` was a docstring example that calls the loss
  directly, which dispatches either way, and the tutorial's training sits in a plain ```julia
  fence that Documenter renders and never executes.

  `test/losses/reduced_loss_optimization.jl` closes that hole — it trains a reduced integrator
  through the functor, and it was checked to fail, with exactly that message, when the annotation
  is put back.

- **Twenty-three of the 25 entry points run to completion, where seven did.** `scripts/` has 25 entry
  points — two under `verification/` and 23 under `reproduction/` — and the survey above found
  seven of them completing, five still computing at the ceiling and 13 failing. Each
  failure was found by the new CI gate rather than by reading, and each cause was measured. The
  classes that recur across several files:

  - **`save` and `load` are not exported names any more.** Seven scripts called a bare `save` and
    one a bare `load`; they mean `CairoMakie.save` for a figure and `JLD2.save`/`JLD2.load` for
    weights.
  - **Eight scripts write into an output directory they do not create** — `comparison_plots/`,
    `phase_space_samples/`, `plots/`, `abc_flow/`, `rigid_body/` and
    `symplectic_autoencoder_validation/`. Each now calls `mkpath` first. This is the class that
    cannot be found locally: once any earlier run has made the directory, every later run passes,
    and only a fresh checkout fails. Four of them surfaced as gate failures; the other four were
    found by reading the save sites of the scripts those four sat beside.
  - **Two scripts built a `Float64` network against a `Float32` `DataLoader`.** The networks now
    name their element type.

  The rest, one by one:

  - `scripts/utilities/convert_jld2_to_h5.jl` **corrupted `docs/Project.toml`.** It activated the
    documentation environment and called `Pkg.develop(path = "..")` on it, which writes a
    machine-local absolute path into that file's `[sources]` table — where a relative `{path =
    ".."}` already stood. It runs in the scripts environment now, which has the three packages it
    needs, and activates nothing.
  - `scripts/verification/network_parameters_gradient_projection.jl` could not load:
    `ChainRulesCore` and `NeuralNetworkParameters` were not in `scripts/Project.toml`. Both are now.
  - `scripts/reproduction/symplectic_autoencoders/training.jl` called
    `AdamOptimizer(η, β₁, β₂, δ)`. `Adam` dropped the `η` field it never applied to the direction,
    and made the other three keywords precisely so that the old positional call fails instead of
    silently reading `η` as `β₁`. The learning rate is the `Optimizer`'s `step_size` now.
  - `scripts/reproduction/symplectic_autoencoders/plot_waves.jl` wrote into a `plots/` directory
    that does not exist, and read a snapshot matrix it never said anything about producing. It
    creates the directory, and runs `integration.jl` itself when the matrix is absent.
  - `scripts/reproduction/symplectic_transformer/double_pendulum_phase_space_plot.jl` and
    `double_pendulum_short_integration.jl` imported `timespan` and `timestep` from
    `GeometricProblems.DoublePendulum`, which declares neither: they are `DEFAULT_TIMESPAN` and
    `DEFAULT_TIMESTEP`. The sibling `double_pendulum.jl` was repaired earlier in this release; these
    two carried the same defect and were not in that sub-task's list.
  - `double_pendulum_short_integration.jl` also had `const` on a local, which is a lowering error
    that stops the whole file, and a `dl` that the same function assigned after using the outer one
    — so the repaired file would have failed on an undefined local. The inner one is `dl_short`.
  - `scripts/reproduction/sympnets/sympnet_pendulum.jl` indexed its data from zero. `pendulum_data`
    returns `1 × n_time_steps` matrices and once returned `OffsetArray`s; the script now flattens
    them with `vec` and counts from one, and its loss draws the predicted index from `2:ntime`
    rather than `1:ntime`, since it reads the point before.
  - `scripts/reproduction/volume_preserving_feedforward/abc_flow.jl` loaded `Metal`, which is not a
    dependency and does not install on the Linux runner, and then built `Batch(batch_size, 1)` — a
    `Batch{:Transformer}` of sequence length one — for a `NeuralNetworkIntegrator`, whose loss
    defaults only for a `Batch{:FeedForward}`. It runs on the CPU, with the Apple-GPU setup it was
    written for recorded in a comment beside the backend constant. **It reproduces its figure in
    `Float64` on the CPU now, where the committed one was made in `Float32` on the GPU.**
  - `scripts/reproduction/sympnets/sympnet_pendulum.jl` was three API generations behind.
    `Gradient(n, upscaling, activation; change_q)` split into `GradientLayerQ` and
    `GradientLayerP`; `Lux.Chain` rejects those with "Encountered a non-AbstractLuxLayer in Chain",
    because they are `AbstractNeuralNetworks` layers, so the container is the `Chain` this package
    exports and the network applies to `(input, parameters)` with no state to thread; and
    `optimization_step!` takes `GlobalSection(ps)` where the script passed the model. `using Lux`
    goes with them — nothing in the file needed Lux once the container changed, and while both were
    loaded a bare `Chain` was ambiguous and resolved to nothing.
  - **`scripts/reproduction/symplectic_autoencoders/training.jl` is rewritten onto the current
    API, and its reduction-error study runs again.** It was the furthest behind of any script here,
    and every layer only became visible once the one above it was repaired: `AdamOptimizer`'s
    signature, `PSDLayer`'s `retraction` keyword (the retraction is the `Optimizer`'s now),
    `GradientQ`/`GradientP`, `initialparameters`' signature, the bare `loss` that is
    `AutoEncoderLoss`, its output directory, and then the two structural ones below.

    Its **training** was a hand-rolled loop over `redraw_batch!(dl)` that counted its own iterations
    from `dl.batch_size`. A `DataLoader` carries neither: the batch is a `Batch` and the loop is the
    `Optimizer` functor. That loop also wrapped both the pullback and the optimization step in
    `try … catch; continue`, so a run in which *every* step failed was indistinguishable from one
    that worked — which is why the script could be this far out of date without anyone noticing.

    Its **reduced systems** were built from closures over hand-sliced parameters. `ReducedSystem`,
    `Symplectic`, `perform_integration_full`, `perform_integration_reduced` and
    `reduced_vector_field_from_full_explicit_vector_field` are all gone, and the replacement
    `HRedSys` takes a `NeuralNetwork{<:SymplecticEncoder}` and a
    `NeuralNetwork{<:SymplecticDecoder}`. So the reductions are neural networks now: `PSDArch` with
    `solve!` for the proper orthogonal decomposition the script computed by hand with `svd`, and
    `SymplecticAutoencoder` for the encoder/decoder chain it spelled out layer by layer.
    `integrate_full_system`, `integrate_reduced_system`, `reduction_error` and `projection_error`
    replace the four retired entry points.

    One thing changes shape. The full solution came from a `ReducedSystem` built with `nothing` for
    both encoder and decoder; `HRedSys` will not take that, so the first reduced system of each `μ`
    sweep supplies it. It is the same quantity — the full system does not depend on the reduction —
    computed once per `μ` as before.

    **It is not deleted in favour of `online_sympnet.jl`, because that script does not cover it**:
    a different problem (the parametrised wave equation against the Toda lattice), a sweep over
    fourteen reduced dimensions and four parameter values against one of each, and the PSD-versus-
    autoencoder projection and reduction error that is this script's whole product, which
    `online_sympnet.jl` has commented out. The run produces its four `plots/v3mu*.png` again.
  - Both `scripts/reproduction/symplectic_autoencoders/online_*.jl` get their element types, their
    backend switch, their `JLD2.save`/`CairoMakie.save` qualifications and, for
    `online_transformer_for_sae.jl`, an explicit statement of the weights dependency it has on
    `online_sympnet.jl`. Both then ran once the `ReducedLoss` annotation above was removed; neither
    is skipped.

- **The reproduction scripts call `default_parameters()` rather than passing the name.**
  `GeometricProblems` defines `default_parameters(::Type{T} = Float64) where {T}` in each problem
  module, so the bare name is the function itself, not the parameter `NamedTuple` the scripts want.
  Fourteen sites across seven files in `scripts/` now call it. The sites under `docs/src/tutorials/`
  were already correct.

- **The Hamiltonian passed to `plot_result` in `scripts/reproduction/harmonic_oscillator.jl` takes
  its arguments in the right order.** `HarmonicOscillator.hamiltonian` is
  `(t, q, p, params)`, but `H` passed the momentum slice as `t`, `0.0` as `q` and the whole state
  vector as `p`, so every call was a `MethodError` and no contour was ever drawn. `H` now evaluates
  to `p² / 2m + k q² / 2`, which is what `plots.jl` expects to contour over the phase space grid.

- **Three reproduction scripts run again; two more get one blocker removed but still fail.**
  `GeometricProblems.DoublePendulum` no longer defines `timespan` or `timestep` — the values are
  `DEFAULT_TIMESPAN` and `DEFAULT_TIMESTEP` — so
  `scripts/reproduction/volume_preserving_feedforward/double_pendulum.jl` and
  `scripts/reproduction/symplectic_transformer/double_pendulum.jl` import `DEFAULT_TIMESPAN`
  instead. The first of the two also dropped an unused `hamiltonian` from that import. Both
  `CairoMakie` and `GeometricMachineLearning` export `save`, which makes the bare name ambiguous, so
  `scripts/reproduction/sympnets/sympnet_toda_lattice.jl` and
  `scripts/reproduction/symplectic_transformer/double_pendulum.jl` write `CairoMakie.save`. The
  second wrote into a `comparison_plots/` directory that nothing creates, and now calls `mkpath`
  first. Those three now run to completion, or past two minutes of training without error.

  The two `scripts/reproduction/sympnets/sympnet_pendulum*.jl` included a `pendulum.jl` beside them,
  where the file is `scripts/utilities/pendulum.jl`. Correcting the path only moves their failure:
  both now abort on their first `lines!` call, because `pendulum_data` returns 1×N matrices and
  Makie wants vectors. Further back they still name `Gradient`, which this package no longer
  exports, and `Chain`, which is ambiguous with `Lux`'s. They are stale against an API several
  releases old and need a rewrite, not a repair. **Neither runs.** The CUDA variant fails at the
  same `lines!` call, before it reaches any CUDA code, so this is not about absent hardware.

- **The double pendulum validation problem got the wrong keyword, and its training the wrong batch.**
  `GeometricProblems.DoublePendulum.hodeproblem` takes `parameters`, and
  `scripts/reproduction/volume_preserving_feedforward/double_pendulum.jl` passed `params`, so the
  corrected `default_parameters()` above never reached it. The same script built its feedforward
  batch with `Batch(batch_size, 1)`, which is a `Batch{:Transformer}`; an `Optimizer` call on a
  `VolumePreservingFeedForward` network has no method for that, and the training threw a
  `MethodError`. It is `Batch(batch_size)` now.
  `scripts/reproduction/volume_preserving_feedforward/rigid_body.jl` built the same wrong batch for
  the same kind of network, and is corrected with it. That script needs a CUDA driver to reach the
  call, which is why the fault survived this long.

- **The tensor Cayley kernels are checked for orthonormality in `Float32`.**
  `test_tensor_cayley2` through `test_tensor_cayley5` each took a `T::Type` argument and then
  ignored it: every one built its input with `rand(n, n, third_dim)`, which is always `Float64`.
  `check_all(Float32)` therefore ran exactly the same arithmetic as `check_all(Float64)`. The four
  now build with `rand(T, n, n, third_dim)`.

  What this adds is the orthonormality assertion in single precision, and for the 5×5 this file
  remains the only place the kernel is exercised at all.
  `tensor_cayley2` through `tensor_cayley4` did already run in
  `Float32`: `test/attention_layer/attention_setup.jl` sends `Float32` parameters through
  `VolumePreservingAttention` at sequence lengths 2, 3 and 4, which is what selects those three
  kernels. Those tests assert volume preservation and parameter element type, never `B * B' ≈ I`.
  They never reach `tensor_cayley5` at all — their fourth case is sequence length 10, which takes
  the generic `cpu_tensor_cayley` branch instead.

  The kernels pass the orthonormality assertion in single precision with a wide margin, and no
  tolerance was touched. Over 40 000 random slices per size, the quantity `B * B' ≈ one(B)` actually
  compares — `norm(B * B' - I)` against `rtol * max(norm(B * B'), norm(I))`, with
  `rtol = sqrt(eps(Float32))` = 3.45e-4 — reaches at most 0.07% of what it is allowed for the 2×2,
  0.12% for the 3×3, 0.14% for the 4×4 and 0.15% for the 5×5. The kernels are also type-preserving:
  a `Float32` input gives a `Float32` result rather than silently promoting.

  `check_all` calls all four sizes, so every Cayley kernel is now exercised in both precisions.
  `check_orthonormal_property` takes an `AbstractArray` rather than an `Array`, so a GPU backend
  can reuse it.

- **The `*_inverse_pullback` tests checked one slice ten times.** All five loop `for i in 1:k` but
  indexed every slice with `k` — the slice count, which is the function's own argument, not the loop
  variable — so each iteration re-compared the last slice. The `rrule`s for `tensor_inverse2`,
  `tensor_inverse3`, `tensor_inverse4` and `cpu_inverse` were asserted on a tenth of the slices they
  named; `tensor_inverse5`'s on none, its two entries being commented out. The indices are `i` now,
  and all five pass slice by slice, the 5×5 pair included — the kernel regeneration entry above
  enables it.

  Two consequences of the wrong index go with it. The per-slice pullback is `pullback_i` rather than
  `pullback_k`, a name that was accurate only while the loop compared slice `k`. And the whole-tensor
  cotangent is computed once into `in_diff` above the loop rather than rebuilt on each of the ten
  iterations, nine of which were redundant.

- **The shape of the optimizer cache no longer depends on which types `GeometricOptimizers` happens
  to accept.** `_make_optimizer_cache` and `_make_optimizer_state` asked the capability question
  (`x isa GeometricOptimizers.OptimizerSolution`, via `_use_go_cache`) *before* the structural one, so
  a `NetworkParameters` reached the container branch only because it is not currently a member of that
  union. The moment `GeometricOptimizers` adopts the container — which is the next thing it does — the
  root of a network would have matched `_use_go_cache` instead, and a whole network would have been
  given one cache rather than one per layer, with `_leaf_optim_step!` handed the entire tree and
  `_GMLGradient` handed a `NetworkParameters` it has no method for. A `MethodError` on the first step
  of every training run, from a change that reads as purely additive upstream.

  The `NetworkParameters` branch now comes first. The `NamedTuple` branch deliberately stays *after*
  `_use_go_cache`, and that asymmetry is the fix rather than an oversight: a layer is a `NamedTuple` of
  arrays, which is exactly what one `GeometricOptimizers` cache is for, so hoisting it too would
  descend into the individual weights. Behaviour today is unchanged, which is what makes it safe to
  land before the upstream release rather than with it.

  `_tree_optim_step!` had the same inversion in its descent into `λY`, and it is fixed the same way:
  the test now names both container types, because a tree of sections is something to descend into
  whichever type carries it. Today only a `NamedTuple` arrives, since this package's own
  `GlobalSection(::NetworkParameters)` unwraps the container first — but that method is the one it
  gives back next, and its replacement upstream returns a container, at which point a single
  `isa NamedTuple` would have handed every layer the whole tree instead of its own section.

  `test/optimizers/utils/optimization_step.jl` is the regression net, and pins the shape rather than
  the run: a network's cache and state are `NamedTuple`s keyed by its layers, with one
  `GeometricOptimizers` cache and one state per layer — not one for the root, and not one per weight.
  `test/optimizers/structured_array_parameters.jl`, four architectures × four methods, covers the step
  itself.

- **`test/runtests.jl` ran the tensor-inverse tests twice.** It included
  `test/kernels/tensor_inverse.jl` from two `@safetestset` blocks — `"Custom inverse for 2x2, 3x3,
  4x4, 5x5 matrices"` and `"Test parallel inverses"` — with no GPU guard, no version guard and no
  other condition on either, so a full `Pkg.test()` evaluated the same 140 assertions 280 times for
  no added coverage. `"Test parallel inverses"` is the one that stays: it entered with the two kernel
  test files themselves, alongside its `"Test parallel Cayley"` sibling, and still sits next to it.
  `"Custom inverse for 2x2, 3x3, 4x4, 5x5 matrices"` arrived four days later inside an unrelated
  data-loader commit, which added a second `include` of `test/arrays/triangular.jl` in the same
  breath — a slip, not a second configuration, and the two halves of the file were never a CPU/GPU
  split. `test/kernels/tensor_cayley.jl` was included once already, and no `include` in
  `test/runtests.jl` is repeated now.

- **`git status` no longer reports the data file that a script generates.**
  `scripts/reproduction/symplectic_autoencoders/integration.jl` writes a snapshot matrix of about
  7.9 MiB at its full size, and still writes one on every run. `.gitignore` did not cover it, so the
  working tree came back dirty each time and the artefact sat one staging sweep away from entering
  the history. It is ignored now; deleting it is still a manual step.

  Where it lands depends on how the script is started: the file name reaches `h5open` as a bare
  relative string, so it is written to the process working directory rather than beside the script.
  Started from the repository root it appears at the root; started from
  `scripts/symplectic_autoencoders/`, it appears there. Both invocations are covered — the root by
  an anchored `/snapshot_matrix*.h5`, which also takes in the `snapshot_matrix2.h5` named by the
  script's `p_zero` branch, a branch not taken as committed; the script's own directory by
  `scripts/**/*.h5`.

  Neither pattern matches the six committed network weights under `docs/src/tutorials/` —
  `sae_parameters.h5`, `integrator_parameters.h5`, `integrator_parameters_psd.h5` and
  `transformer_rigid_body_nn_{st,vpff,vpt}.h5` — which the tutorials load in `@setup` and `@example`
  blocks instead of retraining. Keeping the patterns off that directory is what leaves room for a
  *new* weight file to be committed there: a bare `*.h5` would leave the six already-tracked files
  alone, but would make `git add` refuse a seventh.

### Added

- **`LagrangianNeuralNetwork` is trainable through `DataLoader` + `Batch` + `Optimizer` now.**
  `LNNLoss` is its loss and `NetworkLoss(::LagrangianNeuralNetwork)` returns one, so the architecture
  no longer reaches training only through the `train!` harness. The network output is a scalar
  Lagrangian ``L(q, \dot{q})``; the loss solves that Lagrangian's Euler–Lagrange equations for the
  acceleration and compares the result against the acceleration in the data, relative to the norm of
  the data, the way `HNNLoss` compares a vector field. `input` stacks ``q`` on ``\dot{q}`` and has
  ``2n`` rows; `output` is ``\ddot{q}`` and has ``n``.

  **This repairs the training method it replaces rather than transcribing it.** `loss_single` in
  `src/training_method/lnn_exact_method.jl` reads `abs(sum(∇q∇q̇L(nn, qₙ, q̇ₙ, params)))`, and it
  takes a `q̈ₙ` argument it never uses: sweeping that argument over `-5`, `0` and `5` returns
  `0.291075` every time. It is a penalty on the mixed Hessian block, minimised by any Lagrangian
  whose position and velocity do not couple, and it fits no data at all. What the method meant to
  compute is the rest of that same line, commented out — and that expression is incomplete too,
  contracting with neither ``\dot{q}`` nor ``\ddot{q}``. What `LNNLoss` uses is the Euler–Lagrange
  equation written out:

  ```
  ∇q̇∇q̇L q̈ + (∇q∇q̇L)' q̇ = ∇qL
  ```

  The transpose is load-bearing, because ``\nabla_q\nabla_{\dot{q}}L`` is indexed
  ``[i, j] = \partial^2L/\partial{}q_i\partial\dot{q}_j`` while the chain rule contracts the first
  index. `test/lagrangian_neural_network_tests.jl` checks the solve against a closed form for
  ``L = \tfrac12\dot{q}^TM\dot{q} + q^TC\dot{q} - \tfrac12q^TKq`` with `C` deliberately *not*
  symmetric, and also checks that the same expression with the transpose dropped does **not**
  reproduce that closed form — so the assertion cannot pass for a reason unrelated to the index
  convention.

  **The derivatives come from a compiled symbolic expression, not from `Zygote` inside the loss.**
  Differentiating a loss that itself calls `Zygote.gradient` with respect to the network parameters
  fails with `MethodError: no method matching getindex(::IdDict{Any, Any})`; measured in two separate
  processes. `SymbolicNeuralNetworks.Jacobian` applied to its own result gives the input-Hessian
  instead, and it agrees with `Zygote.hessian` to `2.8e-17`. This is the route
  `hamiltonian_vector_field` already took, and this is why.

  The solve inverts ``\nabla_{\dot{q}}\nabla_{\dot{q}}L``, so its conditioning is what decides
  whether the solved form is usable at all rather than an un-inverted residual. It is asserted in
  the suite, not merely measured once: `test_velocity_hessian_is_well_conditioned` draws fresh
  parameters and fresh evaluation points and requires the condition number of that block to stay
  below `1e8`. On the draws behind the choice it never came close — median between `1.0` and `7.9`,
  worst `3332`.

  What is **not** added is a `SymbolicPullback` for this architecture. `HNNLoss` has one, and *B5*
  under *Open Issues* records why that machinery is not trustworthy for a loss that is not additive
  over the batch — which this one, dividing by `norm(output)`, is not.

- **`SymplecticEulerLoss` and `VariationalMidpointLoss` carry the training methods' numerical content
  onto `NetworkLoss`.** Of the four methods the `train!` harness held that had no modern counterpart,
  two are ported here, one is judged not worth porting, and the fourth — `HnnExactMethod` — was
  already `HNNLoss`. These three ports landed before the deletion under *Removed (breaking)* above,
  so the subsystem leaves with its numerical content already carried over rather than lost with it.

  `SymplecticEulerLoss(arch, timestep)` trains a Hamiltonian neural network on a *trajectory*.
  `HNNLoss` needs ``(\dot{q}, \dot{p})`` in the data; this one needs two consecutive states and the
  timestep between them, and asks that one symplectic Euler step of the learned Hamiltonian carry the
  first state to the second. Variant `:A` evaluates the vector field at ``(q_{n+1}, p_n)`` and `:B`
  at ``(q_n, p_{n+1})``, which is the whole difference between the two methods it replaces.

  **The manual has described this loss since before it could be run.**
  `docs/src/architectures/hamiltonian_neural_network.md` has a section *HNN Loss for Phase Space
  Data* giving the formula with the field at ``(q^{(t)}, p^{(t+1)})`` — symplectic Euler B — and the
  only implementations of it were `SEulerA`/`SEulerB`, which raise a `MethodError` on `vectorfield`
  before computing anything. The page now points at a loss that runs.

  `VariationalMidpointLoss(arch, timestep)` trains a Lagrangian neural network on positions alone,
  through the discrete Euler–Lagrange equations
  ``D_2L_d(q_n, q_{n+1}) + D_1L_d(q_{n+1}, q_{n+2}) = 0`` of the midpoint discrete Lagrangian. It
  needs neither velocities nor accelerations in the data.

  **`BasicSympNetMethod` gets no port, and nothing is lost by that.** It goes with the rest of the
  subsystem under *Removed (breaking)* above, and it is the one method whose content does not need
  carrying over first, because the package already has it. Its `loss_single` is
  `sqeuclidean(q̃ₙ₊₁, qₙ₊₁) + sqeuclidean(p̃ₙ₊₁, pₙ₊₁)` on a one-step prediction — which is
  `FeedForwardLoss`, unnormalised and squared. Measured on one network and one pair of states, the
  method returns `sum(abs2, prediction - target)` and `FeedForwardLoss` returns
  `norm(prediction - target) / norm(target)`. It also only ever ran at one degree of freedom:
  `q̃ₙ₊₁, p̃ₙ₊₁ = nn([qₙ..., pₙ...], params)` destructures the output vector into two scalars, so at
  two degrees of freedom it raises `DimensionMismatch: first collection has length 1 which does not
  match the length of the second, 2`. A SympNet trained through `DataLoader` + `Batch` + `Optimizer`
  already uses `FeedForwardLoss`.

  **Neither ported method could be ported by transcription, because neither ran.** `SymplecticEulerA`
  and `SymplecticEulerB` are *B6*: both call `vectorfield`, which is defined nowhere.
  `hamiltonian_vector_field` is what they meant, and it is what the loss uses.
  `VariationalMidPointMethod` failed for two reasons of its own. `discrete_lagrangian` returns the
  network output, an array, so `Zygote.gradient` refuses it outright — "Output is an array, so the
  gradient is not defined". And `DL₁`/`DL₂` slice the 2-tuple of gradients `DL` returns with
  `1:length(qₙ)` and `(1 + length(qₙ)):end`, selecting *tuple entries* rather than vector components,
  which happens to coincide with the intent at one degree of freedom and is wrong at every other.

  **The derivatives come from compiled symbolic expressions, not from `Zygote` inside the loss.**
  Differentiating a loss that itself calls `Zygote.gradient` with respect to the parameters fails
  with `MethodError: no method matching getindex(::IdDict{Any, Any})`, measured in two separate
  processes. For the variational loss this is avoided by writing the chain rule through the midpoint
  out by hand — ``D_1L_d = \tfrac{\Delta{}t}{2}\nabla_qL - \nabla_{\dot{q}}L`` and
  ``D_2L_d = \tfrac{\Delta{}t}{2}\nabla_qL + \nabla_{\dot{q}}L``, both at the midpoint — so that only
  the network's first input derivative is needed. `test/training_method_losses.jl` checks those two
  expressions against a `Zygote` gradient of ``L_d`` itself, which is the call they exist in order
  not to make.

  Each loss is asserted to vanish on a trajectory that satisfies it and to *not* vanish on a
  perturbed one, so neither assertion can pass for a reason unrelated to the method. One limit is
  worth stating: the discrete Euler–Lagrange equations of an arbitrary neural-network Lagrangian are
  a root-finding problem with no guaranteed solution near a given starting pair, so the variational
  test searches for a starting pair that admits one instead of assuming that any will.

- **`test/exports.jl` asserts that every name the package exports is actually defined.** Julia only
  errors on a dangling `export` when the name is *resolved*, so an exported name that nothing defines
  is silent: the package loads, the docs build and the suite passes, and a user who reaches for the
  name gets an `UndefVarError` from something the package advertises. Ten such names are in the
  export list today — `CPUDevice`, `Device`, `LinearSymplecticLayerP`, `LinearSymplecticLayerQ`,
  `ResidualLayer`, `aresame`, `convert_to_dev`, `description`, `symbol` and `timestep` — and each
  carries a short reason in the allowlist, so the ten stay visible instead of being rediscovered.
  This is the assertion *C10* under *Open Issues* names as the fix; the model is
  `GeometricOptimizers`' `test/exports.jl`.

  The ten are not one case. `ResidualLayer`'s definition exists, at `legacy/layers/resnet.jl`, which
  nothing under `src/` includes — the loaded layer of that shape is `ResNetLayer`. `description` and
  `timestep` are `GeometricBase` generics that package defines but does not export, so each needs an
  `import` rather than a definition. `Device`, `CPUDevice` and `convert_to_dev` have call sites under
  `test/performance_tests/`, which *C11* records as unreachable from `runtests.jl`.

  Nothing about the package's surface changes — the ten are still exported and still undefined, and
  deciding each one (define it, or drop the export) remains to be done. What changes is that the
  class is closed in both directions: an eleventh cannot be added without the suite saying so, and
  an allowlist entry whose name later resolves, or stops being exported, fails until it is removed.
  So the ten cannot be decided one by one and their stale reasons left behind.

- **`scripts/test_attention.jl` and `scripts/test_double_multiplication_derivative.jl` are testsets
  under `test/` now, not scripts nothing ran.** Both failed identically on evaluating `ps.params`:
  `NetworkParameters` overloads `getproperty` to reach into the wrapped `NamedTuple`, so `.params`
  looks up `:params` as a key of that `NamedTuple` instead of returning it, and raises a
  `FieldError`. The accessor for the wrapped `NamedTuple` is the free function `params(ps)` —
  imported by this package but not exported.

  Fixing that access was enough to run both, and what they actually probe, once running, is
  *structure preservation through Zygote*, not a derivative identity: a gradient taken with respect
  to a `NetworkParameters` wrapper comes back as a `NetworkParameters` whose `SymmetricMatrix` leaf
  is preserved when the loss expression performs a single `getproperty` on that wrapper
  (`symplectic_attention_simplified`, `symplectic_linear_map`, `single_multiplication`), and
  degrades to a plain `Matrix` on the second access (`symplectic_attention`,
  `double_multiplication`). The `SymplecticAttentionQ` layer's own forward pass also keeps the
  structure, but by way of a second route described below.

  **The count that decides this is of accesses on the wrapper, not of uses of the leaf.** The two
  coincide in every case above, which is what makes the weaker reading easy to adopt and wrong.
  `gradient(p -> (A = p.L1.A; sum(A) + sum(A)), ps)` uses the leaf twice through one access and
  returns a `SymmetricMatrix`; `gradient(p -> sum(p.L1.A) + sum(p.L1.A), ps)` uses it twice through
  two accesses and returns a `Matrix`. On this route accesses *below* the wrapper do not count:
  `p -> (L = p.L1; sum(L.A) + sum(L.A))` reads the inner `NamedTuple` twice and keeps the structure.
  A third access behaves as the second. The same expressions against `params(ps)`, the bare
  wrapped `NamedTuple`, keep the structure at any access count — but not because the reverse pass
  treats them differently. `Zygote.pullback` drops the leaf to a `Matrix` on the second access for
  the bare `NamedTuple` exactly as it does for the wrapper; what differs is what happens after it.
  `Zygote.gradient` finishes by projecting, and `ChainRulesCore.ProjectTo` of a `NamedTuple` is a
  structured projector whose leaf maps the raw `Matrix` back to a `SymmetricMatrix`. `ProjectTo` of
  a `NetworkParameters` falls back to `identity`, because `NeuralNetworkParameters` defines no
  method for it, so nothing restores the wrapper's leaf. `_custom_mul` is not involved either —
  neither expression above calls it. Defining that `ProjectTo` method upstream would close both
  gaps. Until then the two-access cases are `@test_broken`, honestly, rather than deleted or
  asserted to work; every other case asserts the `SymmetricMatrix` survives.

  **`getproperty` is not the only route in, and the other one counts at a different level.** A
  `Chain` reaches its layers through `values(ps)` rather than through `ps.L1`
  (`AbstractNeuralNetworks/src/chain.jl:56`, `applychain(layers, x, ps::NetworkParameters) =
  applychain(layers, x, values(ps))`). On that route the accesses that decide the split are the ones
  on the plain `NamedTuple` the layer is handed, and the wrapper count no longer shields them:
  `p -> (L = values(p)[1]; sum(L.A) + sum(L.A))` returns a `Matrix`, where the `getproperty` form of
  the same expression returns a `SymmetricMatrix`. So "below the wrapper is free" holds for
  `getproperty` and fails for `values`.

  That is what the `SymplecticAttentionQ` case rests on, and it settles what the local binding at
  `src/layers/symplectic_attention.jl` does — which the comment there calls unexplained. The binding
  collapses the layer body's two reads of `A` into one access on that `NamedTuple`, and on the
  `values` route it is load-bearing: an A/B toggling only the binding, same closure, same input,
  same route, returns a `SymmetricMatrix` with it and a plain `Matrix` without it. The comment is
  left in place rather than rewritten, since this measures structure preservation only and says
  nothing about whatever else the binding may have been added for.

  Every case, working or broken, also asserts what the original scripts' "this works" / "this
  doesn't work" comments never actually stated: the gradient taken with respect to the
  `NetworkParameters` wrapper is `isapprox` the gradient taken with respect to the bare wrapped
  `NamedTuple` — measured as exact agreement (`0.0` maximum absolute difference) in every case,
  including the ones that lose the `SymmetricMatrix` structure. Losing the structure is cosmetic
  here, not a wrong gradient.

  One of the scripts' "this works" / "this doesn't work" comments was **inverted**, not carried
  over. `scripts/test_attention.jl` marked the `SymplecticAttentionQ` layer case `# this doesn't
  work`; it works, and the testset asserts that it does. Re-running that line as the script wrote
  it returns a `SymmetricMatrix`, and so does the form the testset uses — the script differentiated
  with respect to a separate hand-built `NetworkParameters` rather than the network's own, and
  neither choice loses the structure. Why the comment said otherwise is not recoverable from the
  script, and nothing in it was ever run under a test. Every other case kept the polarity the
  script gave it.

  `test/attention_layer/symplectic_attention_network_parameters_gradient.jl` and
  `test/custom_ad_rules/double_multiplication_network_parameters_gradient.jl` are wired into
  `runtests.jl` directly; `test/reachability.jl`'s allowlist needs no change, since neither file was
  ever unreachable. The assertion helper both of them call is
  `test/network_parameters_gradient_structure.jl`, included by each rather than copied into both:
  a `@safetestset` is its own module, so sharing means including, and the rule the helper's
  docstring states is the one thing here that must not be allowed to drift between two copies.
  `test/reachability.jl` reaches it through that include, as it does any other shared file.

### Documentation

- `_tree_optim_step!` records why it is *not* written with
  `NeuralNetworkParameters.foreachparameters`, having been an obvious candidate. It walks the **cache**
  tree, which stops at the layer where a cache sits, whereas `foreachparameters` walks the leaf
  protocol and would descend past the layer into individual weights and re-pair every cache with the
  wrong object. And `λY` is broadcast rather than zipped — a single `GlobalSection` may stand in for a
  whole subtree — which `foreachparameters` cannot express, because it takes `values` of each trailing
  argument. The `nothing`-skip is the only thing the two have in common, and it is one line here.

- **The committed `GeometricOptimizers` inventory tracks 0.5.0**, the version this release requires.
  0.5.0 moved the docstrings off the index page onto a dedicated `api/` page, so every one of the 168
  `jl.*` anchors changed — a stale fallback would not have failed the build, it would have silently
  produced 168 links to anchors that no longer exist, which is worse than having no fallback at all.
  The file also gains the `ScalarMomentAdam` family, `NativePade` and `unit_matrix`, and loses the
  `Index` label. All 26 `@extref GeometricOptimizers` targets used here resolve against it unchanged.

  `docs/make.jl` documents regenerating it from the *deployed* inventory of the required version
  rather than from a sibling checkout's `docs/build`, which is only as current as the last local
  docs build — here, one release behind. The old one-liner also called `save` unqualified, and
  `DocInventories` does not export it.

### Infrastructure

- **Aqua runs in the suite, on the six of the eight checks it runs by default that this package
  passes.** `Aqua.test_all` has nine; `undocumented_names` is off by default. It had never run:
  it was in neither `Project.toml` nor `test/`, so the package had no package-level baseline at all.
  `test/aqua.jl` is included from `test/runtests.jl` as `@safetestset "Aqua's package-level
  checks"`, after the subject drivers rather than beside the other two tree-level guards: a
  top-level testset throws when it closes on a failure, so a package-level finding placed first
  costs every subject result in the run. `Aqua` joins `test/Project.toml` at `0.8`.

  `unbound_args`, `undefined_exports`, `project_extras`, `stale_deps`, `deps_compat` and
  `persistent_tasks` all pass on the released Julia versions, reporting `Pass 9, Total 9` between
  them — `deps_compat` emits four assertions and the other five one each. With the piracy gate
  below, the testset totals `Pass 10, Total 10`. **`persistent_tasks` passes** — the known false
  failure of that check comes from a TestRunner shim's one-entry `JULIA_LOAD_PATH`, and a plain
  `Pkg.test()` does not have one.

  **`unbound_args` fails on Julia nightly**, on one method, for one failing assertion there. That is
  the CI matrix's `nightly` job, which is advisory by construction and not a required check. It is
  recorded under *Open Issues* as *B9*, because a figure quoted without its Julia version is a
  figure no reader can match against a red job.

  **`deps_compat` had to be fixed to get there, and the fix is three lines.** `InteractiveUtils`,
  `LinearAlgebra` and `Random` were in the root `[deps]` with no `[compat]` bound. Each is now
  `"1"`, which is what `test/Project.toml` already did for its own stdlibs.

  **`ambiguities` and `piracies` are switched off by name, and not marked `broken = true`.** They
  report 23 and 13. Both counts are recorded under *Open Issues* as *B8* and *B7*, and the 13
  piracies are recorded there **with a witness each** — a call whose behaviour differs between a
  process holding only the owning packages and the same process with this one loaded. Three of them
  change `Base`, so they change every Julia process that loads this package; one turns
  `dim(nn::NeuralNetwork)`'s `MethodError` into a silently returned `nothing`. Fixing them is a
  change to `src/` that this file does not own, and `broken = true` would leave a check reporting
  success while the defects stand — which is the failure mode the two guards below exist to remove.
  Six checks that fail on a real regression are worth more than eight that are all switched off.

  **`Aqua.test_all` needs an enclosing testset, and the one it gets is `runtests.jl`'s
  `@safetestset`.** It wraps each check in a `@testset` of its own and adds no parent, and a testset
  with no parent finalises as soon as it closes — so called bare, the first failing check throws a
  `TestSetException` and the rest never run. That is how the first measurement of this package
  reported `ambiguities` and stopped. `test/aqua.jl` keeps a `@testset` of its own as well, which is
  the parent when that file is run alone, and which is what lets the piracy gate below sit beside
  `test_all` as a second assertion rather than as a top-level `@test` that ends the file when it
  fails.

  **A switched-off check detects nothing, so `piracies` gets a gate.** `test/aqua.jl` asserts
  `length(Aqua.Piracy.hunt(GeometricMachineLearning)) == 13`, which fails when a piracy is added and
  fails when one is removed without the *B7* entry going with it. `ambiguities` gets no such gate:
  17 of its 23 are against methods in ArrayLayouts, FillArrays, Symbolics and GeometricOptimizers,
  so the count moves with those packages' versions rather than with anything in this tree, and an
  exact assertion would turn the suite red on an unrelated upgrade.

- **Test files are guarded for inclusion completeness.** `test/reachability.jl` verifies that every
  `.jl` file under `test/` is either in the transitive `include` closure of `test/runtests.jl` or
  named in an `ALLOWED_ORPHANS` allowlist with a one-line reason. The closure is read off the parsed
  syntax tree, so an `include` that is commented out — with `#` or with `#= … =#` — or that appears
  inside a string literal is not a live include, without the guard having to re-implement Julia's
  lexer to say so.

  Unreachable test files are never run, and such files have been repaired by hand from time to
  time and regressed in silence. Adding a new test file without wiring it in now fails the suite.
  `test/runtests.jl` runs the new testset first, as `@safetestset "Reachability of every file under
  test/"`.

  The allowlist is closed in both directions, as `test/exports.jl`'s is: an entry whose file is
  later wired into `runtests.jl`, or deleted, fails until the entry goes with it. So the 40 cannot
  be worked through one at a time and their stale reasons left behind.

  At the time of writing, `test/` holds 103 `.jl` files: 63 reachable from `runtests.jl`, 40
  unreachable. All 40 are seeded into the allowlist, and each was run on its own in the test
  environment so that its reason states what the file actually does rather than what its name
  suggests. Only **two** still run and assert anything —
  `custom_ad_rules/matrix_vector_multiplication.jl` and `transformer_related/transformer_setup.jl`.
  Of the rest, 23 stop at `using` a package that is neither a dependency nor a test target (14
  CUDA, 6 Lux, and one each of GPUArrays, Flux and OffsetArrays), 13 stop on a name or method the
  package no longer provides — among them `Attention`, `GradientQ` (the layer is `GradientLayerQ`),
  `SymplecticStiefelManifold`, `Rfac`, `timestep`, a non-existent `src/optimizers/householder.jl`,
  and `GeometricProblems`' `default_parameters` after it became a function — and two load without
  error while asserting nothing: `orthogonalization_procedures/gram_schmidt.jl` defines two test
  functions and calls neither, and one Zygote timing loop runs for minutes without an assertion.

  The walk follows `include` calls whose argument is a string literal. An `include` built from a
  variable or an interpolation is not followed; no file in `test/` does that today, and the effect
  if one did would be to report the target as an orphan rather than to pass it silently.

- **The 23 allowlisted files that could not load a required package are deleted, not repaired.**
  `test/performance_tests/` (20 files), `test/cuda/` (2 files) and `test/kernels/vec_add.jl` are
  gone, along with their entries in `test/reachability.jl`'s `ALLOWED_ORPHANS` — 40 entries down
  to 17. None of CUDA, Lux, Flux, GPUArrays or Optimisers is a dependency of this package or of its
  test environment, and no CI job runs on a GPU, so none of these files ever ran there. (`CUDA` and
  `Lux` are dependencies of `scripts/Project.toml`, which these deleted files did not use.)

  What is lost is not test coverage that ran, but coverage that was only ever advertised. 12 of
  the 23 are bare `@time`/`@printf` scratch pads with no `@test` at all. The other 11 —
  `test/cuda/resnet.jl`, `test/cuda/stiefel_manifold.jl`, and 9 files under
  `test/performance_tests/` — do contain `@test` assertions on GPU array types, but every one of
  them loads `CUDA`, `GPUArrays` or `Lux` in its first few lines, well before any `@test`, so the
  load error stops the file before a single assertion runs. None of these 23 files ever exercised
  the checks they contain. GPU coverage, if wanted, is separate work: writing new tests that run,
  against a test environment that can load the packages they need.

- **Nine orphaned test files are deleted from the repository.** These lay unreachable from
  `test/runtests.jl` and appeared in the `ALLOWED_ORPHANS` allowlist. Only one exercises real,
  passing coverage today: `custom_ad_rules/matrix_vector_multiplication.jl` runs 10 assertions and
  all 10 pass — its deletion loses that coverage, and it is deleted anyway because the assertions
  are a ChainRulesCore tutorial example copied verbatim onto a local type, not
  GeometricMachineLearning content. `orthogonalization_procedures/gram_schmidt.jl` only *defines*
  `gram_schmidt_test` and `sympl_gram_schmidt_test` and calls neither, so including it runs no
  computation and checks nothing; nothing is lost by deleting it. It and its four siblings in
  `orthogonalization_procedures/` are remains from when the module's implementation moved to
  `legacy/orthogonalization_procedures/`, leaving test fragments behind — though only three of the
  five have a source counterpart there, and `global_symplectic_section.jl` and
  `symplectic_householder_aux.jl` do not. The remaining seven files
  fail when tested in isolation: `global_symplectic_section.jl` on
  `SymplecticStiefelManifold` (undefined), `householder.jl` on a missing include target
  `src/optimizers/householder.jl`, `symplectic_householder.jl` on `Rfac` (undefined),
  `symplectic_householder_aux.jl` on a missing `PoissonTensor` method for `Float32`,
  `apply_multi_head_attention.jl` on `Attention`, a name the package does not define at all,
  `linear_wave_equation.jl` because its includes depend on OffsetArrays (not in the test
  environment), and `training_phnn.jl` on a `MethodError` in its own locally-defined
  `make_alternative_parameters_by_adding_constant(params::NamedTuple, ...)`: the file passes it
  `GeometricProblems.CoupledHarmonicOscillator.default_parameters` unevaluated, which used to be a
  `NamedTuple` constant of that name and is now a function, so it no longer matches the
  `NamedTuple` dispatch.

  The `test/reachability.jl` allowlist is shortened to match — 17 entries down to 8 — and
  `Pkg.test()` continues to pass. Two of the allowlist's four groups lose their last entry and their
  headings go with them: nothing under `test/` now stops at a package the test environment cannot
  load, and nothing now loads without asserting anything.

- **Two more orphaned test files are wired in, not deleted.** `test/layers/sympnet_upscaling.jl` and
  `test/transformer_related/transformer_setup.jl` are now included from `test/runtests.jl`; their
  `test/reachability.jl` allowlist entries are removed — 8 entries down to 6.

  `sympnet_upscaling.jl` needed two fixes before it could even run. The layers it built were renamed
  from `GradientQ`/`GradientP` to `GradientLayerQ`/`GradientLayerP` at some point, which is the
  `UndefVarError` the allowlist recorded. Its own loop, `for N in 2:2:20, N2 in (2N):2:(4N)`, called
  `test_symplecticity()` with no arguments, so the two loop variables were dead and the same default
  case (`N=4, N2=20`) ran 100 times. Fixing the call to `test_symplecticity(N, N2)` exposed a third
  problem the naming error had always hidden, because the broken loop never reached it: `𝕁 =
  PoissonTensor(N÷2)` compared an `N×N` round-trip Jacobian against an `(N÷2)×(N÷2)` matrix. The fix,
  `PoissonTensor(N)`, is confirmed correct by tracing the layer chain — `PSDLayer(N, N2)` upscales,
  the two `GradientLayer{N2,N2}`s preserve dimension, `PSDLayer(N2, N)` downscales back to `N`, so
  the Jacobian of the whole chain is `N×N`.

  **With those three fixes, the test still failed intermittently at small `N`.** Measured over 150
  trials: 12.7% of random draws at `N=2`, 2.7% at `N=4`, 0% at `N≥6`. This is not floating-point
  noise — the same 150 trials re-run in Float64 gave the same failure rate (11.7%, within sampling
  noise of the Float32 figure) and the same error magnitudes (mean relative error 0.052 in Float32
  vs 0.057 in Float64); Float64 would have crushed a rounding artefact by nine orders of magnitude,
  and did not. It is not conditioning either: `cond(Φ)` for the layer's Stiefel factor is exactly
  `1.0` in every trial, pass or fail, since `Φ` comes from `qr!(...).Q` and is orthonormal by
  construction. The actual mechanism: each layer *is* exactly symplectic to Float32 machine
  precision — the encoder satisfies `E'𝕁_{N2}E = 𝕁_N`, the shear layers satisfy `G'𝕁_{N2}G =
  𝕁_{N2}`, and `PSDLayer`'s symmetric construction gives the decoder the same identity in the other
  direction, `Dec·𝕁_{N2}·Dec' = 𝕁_N` — all three confirmed to hold to within `1e-5` of the
  theoretical identity (measured max deviation `7.8e-7`, at `N=20, N2=80`, over 20 draws at each of
  the 120 swept pairs) — but the round trip
  needs `G` to preserve `P = E𝕁_NE'`, and `P` has rank `N < N2` while `𝕁_{N2}` is full rank: they
  cannot be equal, by a rank argument, regardless of how `Φ` is drawn. Round-trip relative error runs
  `0.011`–`0.275` across the swept `N`, shrinking with `N` but with no reason to vanish at any finite
  `N2`.

  So the test asserts what is actually true — the encoder identity `E'𝕁_{N2}E = 𝕁_N`, the shear
  identity `G'𝕁_{N2}G = 𝕁_{N2}` and the decoder identity `Dec·𝕁_{N2}·Dec' = 𝕁_N`, each to `atol =
  1e-5` — instead of the round-trip composition, which is only ever approximately symplectic and is
  neither computed nor asserted. An earlier version of this fix added `Random.seed!(1234)` to pin
  the round-trip check against this variance; that is removed, since asserting only the exact
  identities leaves nothing to pin.

  The old version also tied the two `PSDLayer` weights, `ps[4].weight.A = ps[1].weight.A`, with the
  comment that the first and last layer must share a weight or they map to a different symplectic
  potential. The tying goes with the round-trip assertion it supported, so its effect is recorded
  here. It is real but not sufficient: over 100 draws at each of three pairs it pulls the round-trip
  deviation down by an order of magnitude — median `3.4e-2` against `4.7e-1` at `(N, N2) = (2, 4)`,
  `5.2e-2` against `7.9e-1` at `(4, 8)`, `3.3e-2` against `7.7e-1` at `(10, 20)` — and the deviation
  still exceeds the `1e-5` tolerance in 299 of those 300 draws. The rank argument above is what
  rules the round trip out, and no choice of weights escapes it. The three layer identities the test
  now asserts each hold whatever the other layers' weights are, so nothing needs tying.

  The testset runs 120 distinct `(N, N2)` pairs — 360 assertions, three per case — in `2m49.1s`.
  Almost all of that is Zygote compiling the three Jacobian closures, which happens once for the
  whole file: with those closures already compiled, the same 120-pair sweep takes `0.3s`. The pair
  count is therefore nearly free, and shrinking the sweep would not make the testset faster.

  `test/runtests.jl`'s testset for `layers/sympnet_layers_test.jl` was labelled "Test symplecticity
  of upscaling layer", though that file only checks tensor-slice consistency — the guarantee it
  advertised was never checked anywhere. It is relabelled "Test tensor-slice consistency of sympnet
  layers", and `layers/sympnet_upscaling.jl`'s new testset carries the actual symplecticity claim as
  "Test symplecticity of the sympnet upscaling layer".

- **`test/` is one directory per subject, and the test environment is `test/Project.toml`.** The
  suite was 15 test files loose at the top of `test/`, four single-file directories (`batch/`,
  `network_losses/`, `parameterlength/`, `optimizers/utils/`) and 62 `@safetestset` blocks written
  out inline in `test/runtests.jl`; a reader looking for the attention tests found them split across
  `attention_layer/`, `transformer_related/` and `volume_preserving_attention/`, with
  `linear_symplectic_attention.jl` loose at the top level. There are now twelve subject directories,
  none holding a single file, and each carries its own driver naming the testsets it runs.
  `test/runtests.jl` is twelve `include` lines and the two guards.

  The drivers are `include`d at top level rather than wrapped in a testset of their own, because
  `@safetestset` expands to a `module` and a module may not appear inside a testset body. The two
  guards stay beside `runtests.jl` rather than moving into a directory: they check the tree rather
  than a subject in it, and `test/reachability.jl` reads `test/` off its own `@__DIR__`.

  **The suite runs exactly what it ran before**: 62 testsets, 4085 passes and 2 `@test_broken`,
  before and after, with every testset name and every per-testset pass count the same. The names
  did lose their trailing padding — every one of the 62 was padded with spaces to align the summary
  column, which the drivers do not reproduce — so the `Test` summary columns are narrower and the
  names are otherwise unchanged. The seven `@info "Starting …"` progress markers are gone with the
  inline blocks; each `@safetestset` already prints its own summary as it finishes.

  Every moved file is a `git mv`: `git diff -M origin/main...HEAD` gives 35 renames, 30 of them at
  100% similarity. Five files carry a content change. Two are an `include` path —
  `parameters/double_multiplication_network_parameters_gradient.jl` and
  `parameters/symplectic_attention_network_parameters_gradient.jl` now include the helper they share
  from their own directory instead of from `../`. Three are a comment naming a path this change
  moved: `parameters/network_parameters_gradient_structure.jl:24`,
  `reduced_order_modeling/sae_error_lower_than_psd_error.jl:9` and
  `parameters/changebackend_tests.jl:5`.

  `test/Project.toml` replaces the `[extras]`/`[targets]` pair in the package's own `Project.toml`,
  following `GeometricIntegrators`. The difference is that the test environment is now exactly what
  it declares. Under `[targets]` it was the package's own `[deps]` plus the extras, so a test file
  could load `ForwardDiff` or `Symbolics` without anyone declaring it for the tests, and the day the
  package dropped that dependency the suite would break for a reason nothing connects to the change.
  It cannot now: `julia --project=test -e 'using ForwardDiff'` fails, and the sixteen `[deps]`
  entries are precisely what the files under `test/` load, derived from their `using`/`import`
  lines. `ChainRulesTestUtils` and `GeometricIntegrators` move their `[compat]` bounds across with
  them, and `SafeTestsets`' bound goes too: all three were bounds on packages the package itself
  does not depend on.
- **`scripts/` is gated by CI, for the first time.** `.github/workflows/Scripts.yml` runs
  `scripts/runscripts.jl`, which runs every `.jl` file under `scripts/verification/` in full and
  every one under `scripts/reproduction/` with `GML_SMOKE` set, each in its own process. Nothing
  had ever run these files: the package's suite loads not one of them, and the figures in the
  manual and the committed weights under `docs/src/tutorials/` were produced by these scripts and
  by nothing else. They were repaired by hand from time to time — "Bring the scripts to the new
  AdamOptimizerWithDecay", "`tspan` -> `timespan`" — and each repair regressed in silence.

  **Run as a tree for the first time since the restructuring began, 34 files gave 14 that ran to
  completion, 5 still computing at a 300 s ceiling, and 15 that failed.** Of the 15, six wanted a
  GPU this machine does not have, one was an include-only helper that was never an entry point,
  and eight were rot.

  The gate then found eleven more failures the survey could not, because a survey with a timeout
  cannot tell "still training" from "works". `transformer_integrator/symplectic_transformer.jl` and
  `symplectic_transformer_vector_softmax.jl` both read as passes at 300 s — they were still
  training. At the smoke size they run past the training and fail on `UndefVarError: save`. Four
  more write into an output directory they never create, which is invisible on any machine where an
  earlier run has made it and fails on every fresh checkout.

  The job is not a required status check: its name does not begin with `Julia `, and the required
  list is static across the tree.

  Three properties of the driver are what make a red run mean something:

  - **Discovery finding nothing fails.** A renamed or emptied `reproduction/` would otherwise give
    a loop with nothing to fail on, and the driver would report success over a gate that had
    stopped running anything.
  - **The per-script ceiling is per mode** — 900 s for a `verification/` script, which runs in
    full, and 300 s for a smoke run, which takes seconds and where five minutes already means
    something is wrong.
  - **A whole-mode budget of 3600 s stops the run and names what it did not reach.** The ceiling
    alone does not keep its own promise: 23 reproduction scripts each entitled to it outlast any
    runner, and the job then dies at `timeout-minutes` with no verdict at all, which is the
    outcome the ceiling exists to prevent.

- **`scripts/` is three directories with stated purposes.** `verification/` holds the checks that
  establish a mathematical claim — `sympnet_upscaling_symplecticity.jl`, which measures *C12*, and
  `network_parameters_gradient_projection.jl`. `reproduction/` holds the runs that produced the
  committed weights and the manual's figures. `utilities/` holds what the other two include, plus
  `convert_jld2_to_h5.jl`.

  The split is what lets the gate have no list of what to run: every file under `verification/` and
  `reproduction/` is an entry point, so a script added to either is gated from the moment it lands,
  and a file that is included rather than run belongs in `utilities/`. `ensemblesolution/plots.jl`
  is what made the rule concrete — run on its own it fails with `UndefVarError: DataLoader`,
  because it was only ever meant to be included from a script that had already loaded the package.
  It is `utilities/ensemble_plots.jl` now, renamed because `utilities/` is flat and `plots.jl` was
  taken.

  `scripts/Script_using_fully_GML/plots.jl` is deleted. It was three lines that included
  `../plots.jl` so that the scripts in that directory could say `include("plots.jl")`; with the
  directory gone, they include `../utilities/plots.jl` directly. `lnn_script.jl` is
  `reproduction/lnn_pendulum.jl`, beside the `hnn_pendulum.jl` it mirrors.

  `README.md`'s example includes `scripts/utilities/pendulum.jl` now.

- **Every reproduction script states two sizes, and CI runs the small one.** `utilities/smoke.jl`
  defines `smoke_size(full, smoke)`, which returns the second when `GML_SMOKE` is set in the
  environment. A script writes `const n_epochs = smoke_size(2048, 2)` and reproduces its result
  when run normally. A smoke run establishes that the script still executes end to end and nothing
  about the result — which is the honest limit of what a runner can check for a job that belongs on
  a GPU.

  Two entry points carry no size constant, and neither costs anything to run.
  `symplectic_autoencoders/analytic_solution.jl` defines functions and computes nothing at top
  level, and `sympnets/sympnet_pendulum_cuda.jl` is in `SKIPPED` below and never runs here.

  Two of the sizes bound an *integration* rather than a training, and both were added late, after
  review found the two scripts still running at their full size on every pull request.
  `symplectic_transformer/double_pendulum_phase_space_plot.jl` integrates an ensemble and saves one
  figure per member — 500 at the full size, four in a smoke run — and
  `symplectic_autoencoders/integration.jl` integrates a 128-site lattice at 20 parameter values,
  where a smoke run does 16 sites at two.

  Because that snapshot matrix now has two shapes, its *name* carries the mode:
  `utilities/snapshot_matrix.jl` returns `snapshot_matrix.h5` or `snapshot_matrix_smoke.h5`, and
  `training.jl` and `plot_waves.jl` ask their `isfile` guard for the one they need. The autoencoder
  weights the two `online_*.jl` scripts cache are named the same way, for the same reason. Without
  it a full run reads whatever an earlier run of the other mode left in the directory, and the
  `isfile` guard is exactly what makes that silent instead of loud. CI never sees this — a fresh
  checkout has no such file — so it is a hazard for whoever reproduces a result locally.

  Four scripts take their *backend* from the same switch: `volume_preserving_feedforward/rigid_body.jl`,
  `volume_preserving_transformer/rigid_body.jl` and the two `symplectic_autoencoders/online_*.jl`
  read `smoke_size(CUDABackend(), CPU())`, so a full run still trains on the GPU it was written for
  and a smoke run exercises the same code on the CPU.

  **Two of the 25 entry points are named in `SKIPPED`, each with the reason it is not run.** That
  list is closed in both directions, as the two test guards' allowlists are: an entry naming a file
  that is no longer an entry point fails the driver. An entry is a backlog item, not a design — it
  says these files do *not* work, not that they are fine.

  - `linear_symplectic_transformer_gpu.jl` pipes its training data through `cu` and then writes
    `CUDABackend()` into three constructor calls; `sympnets/sympnet_pendulum_cuda.jl` hands `lines!`
    the matrices `pendulum_data` returns and stops there, and past that calls `CUDA.device()` and
    `CUDA.zeros` directly, which is what distinguishes it from `sympnet_pendulum.jl`. No CI runner
    has a GPU.

- **`.gitignore` covers the `.jld2` weights and `.pdf` figures the scripts write.** The `.h5` and
  `.png` patterns were added when nothing ran these scripts; the CI job runs all of them on every
  pull request, and without the two new patterns each run left the tree dirty. Nothing tracked under
  `scripts/` has either extension, and both patterns are scoped to `scripts/` so that neither can
  reach the tracked `.pdf` files under `docs/src/assets/` and `legacy/`.

- **Ten dependencies leave `scripts/Project.toml` and two join it.** `BenchmarkTools`, `Distances`,
  `GeometricSolutions`, `KernelAbstractions`, `NLsolve`, `NNlib`, `SafeTestsets`,
  `SymbolicNeuralNetworks`, `Symbolics` and `Test` are declared and used by no file under
  `scripts/` — the earlier entries in this release record several of them as "left declared", and
  this is the change that owns the file. `ChainRulesCore` and `NeuralNetworkParameters` join it
  because `network_parameters_gradient_projection.jl` loads both and neither was declared, which is
  why that script could not run in the scripts environment at all.

- **C5 is removed from *Open Issues*: its premise no longer holds.**
  `.github/workflows/CI.yml` no longer pins an explicit `1.13` job — only `pre` and `nightly` are
  `experimental: true` — resolved by `4281732c` ("Unify the shared GitHub workflows", 2026-08-31),
  which carries no changelog entry of its own. That predates and is unrelated to every sub-task in
  this release's test/script restructuring; this close-out only noticed it, and fixed nothing.

## [0.7.0]

**A layer is wrapped at the `GeometricOptimizers` boundary, and a whole set of parameters is a
`NetworkParameters`.** Breaking: this release tracks `NeuralNetworkParameters` 0.3.0,
`AbstractNeuralNetworks` 0.8.0, `SymbolicNeuralNetworks` 0.8.0 and `GeometricOptimizers` 0.7.0.

### Changed

- **`GeometricOptimizers = "0.7"`.** 0.6.0 is what takes a whole `NetworkParameters` through the
  optimizer, and it was breaking: four `outer!`/`_mul!` methods, one `update!(::BFGSState, …)` and
  `add!` on a parameter set are gone, and two `l2norm` methods moved upstream to `GeometricBase`. None
  of them had a caller here — the `add!` arm was deleted on this side in 0.7 for the same reason.

  0.7.0 goes further and **does** need code here: a whole set of parameters reaches that package only
  as a `NetworkParameters` now, and a bare `NamedTuple` is turned away. This package hands it one
  **layer** at a time, so it cannot convert by wrapping at the root; see the entry below. The full
  suite is green against it, `test/reduced_system.jl` aside — that one needs `GeometricIntegrators`,
  which still pins `SimpleSolvers` 0.12 and is the ecosystem-wide blocker.

- **`ParameterSet` is gone from the signatures here.** `NeuralNetworkParameters` 0.3.0 removes it, so
  `Optimizer` and `optimize_for_one_epoch!` take a `NetworkParameters`, `_get_contents` has a method
  per shape, and the loss functors drop the annotation entirely — it dispatched nothing there, since
  the model and the input/output types settle every method, and both a container and the bare
  `NamedTuple` a reverse pass produces reach them.

  `_tree_optim_step!`'s test for a section tree is now `_is_section_tree`, a named predicate rather than
  an `isa` against a union. It answers one question — descend, or hand this whole section to the layer
  — and naming it is what keeps the container arm from looking like breadth for its own sake.

- **A *flat* set of parameters gets one optimizer cache again, and the branch order is what decides it.**
  `_make_optimizer_cache` asks `_is_layer` — "is this one cache's worth?" — *before* asking whether the
  argument is a tree to descend into. A flat set answers yes to both, and the tree reading is wrong: it
  gives every individual weight its own cache, and for a manifold weight that is not merely wasteful,
  because a bare `Manifold` then reaches `_GMLGradient` where a whole layer should have, which is
  ambiguous against `GeometricOptimizers`' own `Gradient` functor. `_is_layer` accepts a container as
  well as a bare `NamedTuple` for this reason.

- **A layer is wrapped at the `GeometricOptimizers` boundary, and the flatness rule is written out
  here.** `_as_go_solution` wraps a layer in a `NetworkParameters` before handing it to
  `OptimizerCache`, `OptimizerState` or `update!`, and `_leaf_optim_step!` uses the wrapped form
  throughout. The wrap **shares the leaf arrays**, so the step still writes through to the network's
  own weights and nothing is copied back — which is what makes this a boundary change and not a
  change of behaviour.

  What did have to be restated is the test that separates a *layer* from a *subtree*. Upstream made it
  by dispatch: `ArrayNamedTuple{T}` bounded a `NamedTuple`'s values by `AbstractArray{T}`, so a
  `NamedTuple` of branches was not one and got a cache per layer rather than one for the whole of it.
  That alias is gone — it was an alias for `Base.NamedTuple`, so every method on it was a method on a
  `Base` type — so `_is_layer` says it here, where it is this package's rule rather than a property of
  whichever types upstream happens to accept this month.

  **One behaviour follows from the wrap rather than from the old bound.** A layer whose weights do not
  share an element type is now one cache with a promoted `T`, where the `Vararg` bound admitted no `T`
  at all and the layer fell through to one `GMLEuclideanState` per weight — i.e. no manifold handling.
  `_adapt_method_to_T` reads the promotion, so the cache is built for the type the layer actually has.

  `_GMLGradient` gains its `NetworkParameters` method as part of this, and `_gml_rgrad` walks with
  `mapparameters` instead of `Base.map`: it recurses on the branches, so `_gml_rgrad` is only ever
  called on leaves, and it rebuilds in the shape of its first argument — a container, which is what
  `GeometricOptimizers._copyto!(gradient_array(cache), ·)` has a method for.

### Fixed

- **`_GMLGradient` on a bare `Manifold` was ambiguous, and the branch ordering was the only thing
  preventing it.** `_GMLGradient{T}` is a `GeometricOptimizers.Gradient{T}` and `Manifold{T}` is an
  `AbstractMatrix{T}`, so `(::_GMLGradient{T})(::AbstractArray{T})` here and upstream's
  `(::Gradient{T})(::Manifold{T})` are neither of them more specific on the intersection —
  `MethodError: … is ambiguous`, several frames into a solve. Asking `_is_layer` before descending
  keeps a bare `Manifold` from arriving, and that ordering is a correctness fix in its own right, but
  a routing decision must not be the only thing standing between a caller and an ambiguity.

  There is now a method for it, and the answer it gives is this package's: the Euclidean gradient is
  already computed and carried in `dp`, so the projection is `rgrad(x, dp)` rather than upstream's
  `rgrad(x, reshape(grad(vec(x)), …))`, which would evaluate an inner gradient this functor does not
  have. `test/optimizers/gml_gradient_dispatch.jl` pins it, along with the shapes the optimizer
  actually produces.

- **An empty set was one cache's worth.** `all` over an empty collection is vacuously true, so
  `_is_layer(NamedTuple())` was `true` and `OptimizerCache` would have been asked for the element type
  of a set with no leaves to promote. An empty set is *zero* caches' worth and descends.

- `_get_contents` takes a method per shape in its wrapped forms too, rather than a `Tuple` and an
  `AbstractVector` over a union of the two. Same rule as everywhere else in this release.

## [0.6.1]

### Changed

- **Julia 1.11 is the minimum**, up from the 1.10 LTS, in step with the rest of this family of
  packages. Nothing here was accommodating 1.10; the two places that name it —
  `test/training_parameters.jl` and `test/optimizers/optimizer_convergence_tests/psd_optim.jl` — record
  where a measurement was taken and how a defect was found, and both stay as the history they are.

- **`GeometricOptimizers = "0.6"`.** 0.6.0 is what takes a whole `NetworkParameters` through the
  optimizer, and it is breaking: four `outer!`/`_mul!` methods, one `update!(::BFGSState, …)` and
  `add!` on a parameter set are gone, and two `l2norm` methods moved upstream to `GeometricBase`. None
  of them had a caller here — the `add!` arm was deleted on this side in 0.7 for the same reason — so
  the bump is a compat entry and no code. The full suite is green against it.

- **The fifteen sites spelling `Union{NamedTuple, NetworkParameters}` inline now name it.** It is
  `NeuralNetworkParameters.ParameterSet`, added upstream in 0.2.2: the same union, in the package that
  owns the type. `SymbolicNeuralNetworks` had it as `EquationSet`, `AbstractNeuralNetworks` spelled it
  out eight times and `GeometricOptimizers` sixteen; all four name one thing now. Definitionally
  identical, so no dispatch changes anywhere. Compat is `NeuralNetworkParameters = "0.2.2"`.
- **The fifteen sites spelling `Union{NamedTuple, NetworkParameters}` inline now name it.** It is
  `NeuralNetworkParameters.ParameterSet`, added upstream in 0.2.2: the same union, in the package that
  owns the type. `SymbolicNeuralNetworks` had it as `EquationSet`, `AbstractNeuralNetworks` spelled it
  out eight times and `GeometricOptimizers` sixteen; all four name one thing now. Definitionally
  identical, so no dispatch changes anywhere. Compat is `NeuralNetworkParameters = "0.2.2"`.

### Fixed

- **The comment on `_make_optimizer_cache`'s branch ordering had been wrong for a release.** It said
  `NetworkParameters` is not one of the types `GeometricOptimizers.OptimizerSolution` unions, and
  therefore that asking the structural question before the capability question was invisible. 0.5.0 made
  the container a member, so `_use_go_cache` is true at the *root* now and the ordering is load-bearing:
  without it a whole network would get one cache instead of one per layer, silently, with
  `_leaf_optim_step!` handed the entire tree. The code was already right — the ordering was put in
  ahead of the change it anticipated — and only the comment needed correcting. Found in the review of
  [GeometricOptimizers #68](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/68).

  It also records what the better end state is, now that `GeometricOptimizers` takes a whole container
  as one solution: one cache for the network. `_GMLGradient` has its `NetworkParameters` method as of
  this release, but that is one wrapped **layer** and not the tree — the container branch still splits
  the network first. Dropping `_tree_optim_step!` and handing the whole tree to one cache is a change
  of behaviour — one `GlobalSection` tree and one `Q` across every layer instead of one per layer — so
  it still wants its own release rather than being folded in here.

## [0.6.0] — 2026-08-24

**The parameter container moves out to [NeuralNetworkParameters.jl][nnp].**
`AbstractNeuralNetworks` 0.7 took the tree that holds a network's parameters out into its own
package, and removed the old name rather than leaving an alias, so that one type has one name across
the ecosystem. This package follows: what was `NeuralNetworkParameters` is `NetworkParameters`, at
every call site and in the export list. The HDF5 extension follows further — it no longer carries
its own traversal of the parameter set, because the packages that own the pieces now say how each of
them is written and rebuilt.

**Requires Zygote 0.7**, which is the other half of this release. 0.7 stopped unthunking cotangents
eagerly, and thunks reaching this package's `rrule`s exposed three real defects — one of them a
*silently zeroed* Jacobian through `assign_q_and_p`, on every symplectic-autoencoder and PSD path.
The type piracy that had been covering for another of them is gone with it. See *Fixed*.

Three exports go — `RecurrentNeuralNetwork`, `LSTMNeuralNetwork` and `NeuralNetworkParameters` —
and `NetworkParameters` arrives in place of the last. Read *Removed (breaking)* before upgrading.

[nnp]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl

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

  > **Upstream releases.** All four bounds this release tightens resolve from the General
  > registry. Every one of them was registered on 2026-08-23, which is why this release is dated
  > the day after.
  >
  > | bound | version | registered (UTC) |
  > | --- | --- | --- |
  > | `AbstractNeuralNetworks = "0.7"` | 0.7.0 | 04:00 |
  > | `NeuralNetworkParameters = "0.1"` | 0.1.1 | 04:12 |
  > | `GeometricOptimizers = "0.4.1"` | 0.4.1 | 06:10 |
  > | `SymbolicNeuralNetworks = "0.6"` | 0.6.0 | 16:11 |

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

### Infrastructure

- **The GitHub Actions are current, and Dependabot keeps them there**
  ([#248](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/248)). Nothing under
  `.github/workflows/` had been bumped since it was written, and two of the actions had aged past
  working. `julia-actions/cache@v1` speaks a cache-service API GitHub has retired, so every job
  logged `Cache service responded with 400` and CI had been running with **no dependency cache at
  all** — every job rebuilding and re-precompiling the whole tree from scratch. And three of the
  pins were `@latest`, which is not a moving alias but a literal tag that neither upstream still
  moves: `julia-actions/setup-julia@latest` is a November 2024 commit matching no release, and
  `julia-actions/RegisterAction@latest` a November 2022 one, older than that action's own v0.3.2.
  Both read as if they track upstream and do not.

  `actions/checkout` v4 → v7, `actions/upload-artifact` v4 → v7, `julia-actions/setup-julia` v1 →
  v3, `julia-actions/cache` v1 → v3, `codecov/codecov-action` v3 → v7, and
  `julia-actions/RegisterAction` `@latest` → v0.3.2. `julia-buildpkg`, `julia-runtest`,
  `julia-processcoverage`, `julia-docdeploy` and `TagBot` were already on their current major.

  The `arch: x64` matrix pin went with them, which **renames every CI job** — the `- x64` component
  is gone. `macOS-latest` is aarch64, and setup-julia v3 refuses `x64` there unless `force-arch` is
  set, because that build runs under Rosetta, which is not the platform anyone deploys on. No name
  was load-bearing: `main`'s branch protection lists no required status checks.

- **CI resolves the registry over git rather than through a package server** (`JULIA_PKG_SERVER: ""`,
  with `cache-registries: false` so that a restored depot cannot put the staleness straight back).
  The package servers' snapshot of General lags the registry by hours to days:
  `AbstractNeuralNetworks` 0.7.0 was registered at 04:00 UTC on 2026-08-23 and jobs were still
  resolving against a copy that stopped at 0.6.4 hours later, dying in `Pkg.resolve` before running
  anything. It is not a matter of picking a better server — all eight official mirrors, in both
  flavours, and the third-party ones too were serving the identical tree; the lag is at the storage
  server they all pull from.

  This is a workaround, and it is marked as one in `CI.yml`: the cost is the CDN and the registry
  tarball. That comment also records how to check whether it is still needed —
  `curl -sL https://pkg.julialang.org/registries` against
  `gh api repos/JuliaRegistries/General/commits/master --jq .commit.tree.sha`. As of this release it
  still is: the served snapshot stops at `AbstractNeuralNetworks` 0.6.4 and
  `SymbolicNeuralNetworks` 0.5.0, both of which this release's `[compat]` excludes.

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

  (**B1**, **B2**, **B3**, **B4** and **B6** are all closed and their entries are gone: B1 and B2 by
  this release — the duplicated `AdamOptimizerWithDecay` and the split `Manifold`, both under
  *Removed (breaking)* — B3 by SymbolicNeuralNetworks 0.5, B4 by `a427add1`, which repaired the
  documentation build, and B6 by the `train!` retirement in this release: the three methods that
  called the non-existent `vectorfield` are gone, and `SymplecticEulerLoss` carries their content on
  `hamiltonian_vector_field`, with tests that run. The numbers are left vacant rather than reused.)

- **B7. Thirteen methods are type piracy, and three of them change `Base`.** Aqua's `piracies`
  check reports 13, and all 13 are genuine under its definition: the function and every argument
  type belong to other modules. It is switched off in `test/aqua.jl` rather than marked
  `broken = true`, because each of these has a **witness** — a call whose behaviour differs between
  a process with only the owning packages loaded and the same process with this one added.

  The three on `Base` are the severe ones, because they change *every* Julia process that loads
  this package:

  | method | `src` | without GML | with GML |
  |:--|:--|:--|:--|
  | `+(::Float64, ::Tuple{Float64})` | `utils.jl:61` | `MethodError` | `3.0` |
  | `+(::Vector{Float64}, ::Tuple{Float64})` | `utils.jl:67` | `MethodError` | `3.0` — a **scalar**, silently discarding every element but the first |
  | `isapprox(::@NamedTuple{q, p}, ::…)` | `utils.jl:163` | `MethodError` | `true` |

  The first two already carry a `# Type pyracy!!` comment in the source.

  **`dim(nn::NeuralNetwork)` at `src/backends/lux.jl:56` is the most instructive.** The subject is
  the *network*, not the architecture: `dim` on an architecture with no method of its own already
  logs and returns `nothing` upstream (`AbstractNeuralNetworks/src/architecture.jl:8`). `dim` on a
  `NeuralNetwork` is a clean `MethodError` without this package, and with it loaded the call reaches
  the architecture fallback instead, logs an error and returns **`nothing`** — for every user of
  that package, not only for users of this one.

  Three more are functor piracy on `AbstractNeuralNetworks`: `Dense` and `Linear` applied to a
  three-axis array (`src/layers/resnet.jl:63,67,71`), which is a `MethodError` upstream because `*`
  cannot take a 3-tensor. `src/layers/resnet.jl:63` is also one of the ambiguities in *B8*, against
  `Affine`.

  Five are `add!` on `GeometricOptimizers` matrix types
  (`src/arrays/gml_extensions.jl:17,27,32,39` and `:22`). For four of them the upstream generic is
  a `CanonicalIndexError` — it does `x .= a .+ b` and those types have no `setindex!` — so the
  method is load-bearing. `:22`, on `SymmetricMatrix`, is not: that type does support `setindex!`,
  the upstream generic already returns the right answer, and the only observable difference is that
  this one allocates where the upstream allocates nothing.

  **Two of the 13 have no value witness, and that is worth saying plainly.** `:22` above, and
  `add!(C::AbstractVecOrMat, A, B)` at `src/utils.jl:49` — the most invasive of the set, shadowing
  the upstream three-argument `add!` for *every* vector and matrix including
  `AbstractNeuralNetworks`' own internal uses. The value it returns is unchanged. Its witness is an
  allocation regression: it writes `C .= A + B`, materialising the sum, where the upstream generic
  writes `x .= a .+ b` and allocates nothing. One array per call, so the cost scales — measured in a
  fresh process at `--check-bounds=auto`, the minimum of 50 calls is 64 bytes for a 1-element
  vector, 144 at 10, 8256 at 1000, and 112 for a 2×2 matrix, against 0 upstream at every size. Plus
  a guard weakened from `axes` equality to `size`
  equality. A performance regression, not a wrong answer.

  Closing this means deciding, method by method, between deleting the piracy and asking the owning
  package for the method. It is a change to `src/` and it is not small.

- **B8. Twenty-three method ambiguities.** Aqua's `ambiguities` check reports 23, and it is
  switched off for the same reason as *B7*. Seventeen are `PoissonTensor * v`
  (`src/arrays/poisson_tensor.jl:71,74,77`) against left-multiply methods in `ArrayLayouts`,
  `FillArrays`, `Symbolics` and `GeometricOptimizers`. One is the `Dense` functor of *B7* against
  `AbstractNeuralNetworks.Affine`; one is `_GMLGradient` (`src/optimizers/optimizer.jl:21`) against
  `SimpleSolvers.Gradient`; and four are the `HNNLoss`, `LNNLoss`, `SymplecticEulerLoss` and
  `VariationalMidpointLoss` functors against `AbstractNeuralNetworks.NetworkLoss`.

  **A count is not a finding**, and unlike *B7* these have not been triaged for witnesses. That is
  what closing this starts with.

- **B9. One unbound type parameter, which only Julia nightly reports.** Aqua's `unbound_args` fails
  on the `nightly` job over `Base.iterate(nn::NeuralNetwork{<:NeuralNetworkIntegrator}, ics::BT;
  n_points)` at `src/architectures/neural_network_integrator.jl:98`. Its signature is
  `where {T, AT <: AbstractVector{T}, BT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}`, and `T` never
  appears in an argument type — it is reachable only through the bound on `AT`, so dispatch cannot
  determine it and the method can never be called with `T` given explicitly.

  **The check passes on `min`, on `1` and on `pre`, and fails on nightly.** The defect is in the
  signature and is there on every version; what changes is whether Julia's method introspection
  surfaces it. So the assertion counts above are the released-Julia figures, and nightly's testset
  carries one failure among them.

  `nightly` is advisory by construction — `.github/workflows/CI.yml` gives it `experimental: true`
  and job-level `continue-on-error`, and it is deliberately not a required check — so this does not
  gate a merge. It is recorded because a red advisory job with nothing to match it against is a red
  job that the next reader has to re-diagnose.

  Closing it means writing the parameter so that it binds, which is a change to `src/`.

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

- **C9. Seven include sites under `legacy/` name files that do not exist.** Six `legacy/hnn/`
  scripts include `../../scripts/data.jl` and `hnn_simple.jl` includes `../../src/training.jl`;
  neither file exists anywhere in the repository, and neither did before the move to `legacy/`. Of
  the include targets under `legacy/`, the other 21 resolved when this was written — see the
  paragraph below, which is where that number stands now. The spelling is now at least
  consistent with where the files sit, so what remains is a decision about `data.jl`: reconstruct it
  (it generated the pendulum training data, which `scripts/utilities/pendulum.jl` now does) or
  delete the scripts that need it.

  **Eight more of those include sites went stale when `scripts/` was restructured**, across seven
  files: each of the seven `legacy/hnn/*.jl` includes `../../scripts/plots.jl`, and `hnn_lux.jl`
  includes `../../scripts/pendulum.jl` as well. `legacy/hnn/README.md` and
  `legacy/hnn/Project.toml` name the first in prose too, so ten references in nine files. All of
  those paths are under `scripts/utilities/` now.

  They are left alone deliberately, and the reason covers six of the seven files rather than all
  seven. Six already stop earlier, on the `data.jl` above, so repointing them fixes nothing that
  runs. **`hnn_lux.jl` is the exception**: it includes no `data.jl`, both of its includes resolved
  before this release, and neither does now — this is the one file where the restructuring is what
  broke it. It is still left alone, because none of the seven is formatted to this repository's own
  `style = "sciml"` and staging them would have meant reformatting around 340 lines of dead code
  for a one-line change each. The path correction belongs with whatever settles `data.jl`.

  Counting the include sites whose argument is a string literal: 28 under `legacy/`, of which 15
  do not resolve and 13 do. Seven of the 15 were already broken before this release.

- **C12. The sympnet upscaling chain is symplectic layer by layer, not end to end, and whether it
  is meant to be is undecided.** `PSDLayer(N, N2) → GradientLayerQ(N2) → GradientLayerP(N2) →
  PSDLayer(N2, N)`: each layer satisfies its own exact symplectic identity (`E'𝕁_{N2}E = 𝕁_N`,
  `G'𝕁_{N2}G = 𝕁_{N2}`, `D𝕁_{N2}D' = 𝕁_N`), but the composition does not, for an algebraic
  reason rather than a numerical one — the round trip needs the shear pair to preserve the
  rank-`N` embedded Poisson tensor `E𝕁_N E'`, while the pair actually preserves the full-rank
  `𝕁_{N2}`, and a rank-`N` matrix cannot equal a rank-`N2` one for `N2 > N`. Measured directly:
  worst relative round-trip deviation over 20 random chains, `Float64`, weights tied, is `0.0018
  .. 0.84` at `(N, N2) = (2, 4)`, `0.016 .. 0.087` at `(4, 16)` and `0.026 .. 0.072` at `(20,
  40)` — shrinking with `N` but never zero — while the three layerwise identities hold to
  `5.6e-16`–`1.4e-15` at the same sizes. `cond(E) = 1.0` rules out ill-conditioning, and the
  round-trip deviation stays the same order of magnitude in `Float32` as in `Float64` — unlike the
  layerwise identities, which drop with it — which rules out rounding as the cause.

  The check is archived as `scripts/sympnet_upscaling_symplecticity.jl`
  ([#276](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/276)); running it reproduces
  the figures above. What it does not answer is whether exact end-to-end symplecticity was ever the
  intended property of this architecture, as opposed to an approximation that improves with `N` —
  that is a design question for the user, not something this check can settle.

- **C13. Four dead names remain in `src/architectures/lagrangian_neural_network.jl`.** `∇L:30`,
  `∇∇L:35`, `∇q̇∇q̇L:39` and the constant `DEFAULT_LNN_NRUNS:2` have no caller. They were dead
  before this release, which is why it removes only `∇q∇q̇L`, the one its own deletion orphaned.

  `∇L` last had a caller in `7f1b3dd0` (2023-06-22). `∇q̇∇q̇L` appeared only in the commented-out
  remainder of `src/training_method/lnn_exact_method.jl:10`, and `∇∇L` survives as its callee alone.
  `DEFAULT_LNN_NRUNS` was the default `ntraining` of an architecture-local `train!` method added in
  `aa1e0471` and removed in `a27140af`, when the shared harness took that job over. The constant
  stayed behind and has had no caller since 2023-06-08.

  All three functions take their derivatives with `Zygote`, which is the route `LNNLoss` cannot
  use: a nested `Zygote.gradient` inside a loss breaks the parameter gradient. So this is not a
  second route to one answer that a caller might want. Deleting them is the expected decision; it
  belongs to a change that owns this file.

### D. Unverified

Not defects — claims this release makes that nothing has actually checked yet.

- **D4. The upstream fix was measured on one optimizer.** The compile-time figures come from the
  `Adam` path. The quasi-Newton and Newton caches and states were widened on the strength of their
  *inferred types* — a sound argument, but not a measurement. Catalogued upstream as GeometricOptimizers C15.
