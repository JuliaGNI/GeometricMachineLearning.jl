# Known issues

What is known to be wrong in GeometricMachineLearning and is not fixed.

## B. Known defects

### B5 · The symbolic pullback of `HNNLoss` is not the gradient of the batched loss.

- **location:** —
- **kind:** defect
- **found:** 2026-08-16
- **evidence:**
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

  (**B1**, **B2**, **B3**, **B4**, **B6** and **B8** are all closed and their entries are gone: B1 and B2 by
  this release — the duplicated `AdamOptimizerWithDecay` and the split `Manifold`, both under
  *Removed (breaking)* — B3 by SymbolicNeuralNetworks 0.5, B4 by `a427add1`, which repaired the
  documentation build, and B6 and B8 by this release: B6 by the `train!` retirement (the three methods that
  called the non-existent `vectorfield` are gone, and `SymplecticEulerLoss` carries their content on
  `hamiltonian_vector_field`, with tests that run), and B8 by the narrowing of `PoissonTensor`'s `Base.:*`
  to `Strided…` right-hand sides and `Base.getindex` to `Union{Int, Colon, AbstractVector{<:Integer}}`
  indices, reducing method ambiguities from 18 to 1 as asserted in `test/aqua.jl`. The numbers are
  left vacant rather than reused.)

  (**B9** is closed by this release and its entry is gone: the `iterate` method it named binds
  every type parameter, and Aqua's `unbound_args` passes on Julia nightly. See *Fixed* above. The
  number is left vacant rather than reused.)

### B7 · Three methods are type piracy, and closing them is an API change rather than a deletion.

- **location:** `src/layers/resnet.jl:63`
- **kind:** defect
- **found:** 2026-09-19
- **evidence:**
  Aqua's `piracies` check reports 3, down from the 12 this entry opened with; the other nine were
  deleted in this release and are under *Removed (breaking)* above. All three are genuine under
  Aqua's definition — the function and every argument type belong to other modules — and all three
  are one thing: a functor on a *layer* type this package does not own, applied to a three-axis
  array.

  | method | `src` | without GML | with GML |
  |:--|:--|:--|:--|
  | `(::Dense{M, N, true})(::AbstractArray{T, 3}, ::NamedTuple)` | `layers/resnet.jl:63` | `MethodError` | the layer applied along the third axis |
  | `(::Dense{M, N, false})(::AbstractArray{T, 3}, ::NamedTuple)` | `layers/resnet.jl:67` | the same | the same |
  | `(::Linear{M, N})(::AbstractArray{T, 3}, ::NamedTuple)` | `layers/resnet.jl:71` | the same | the same |

  `Dense` and `Linear` are `AbstractNeuralNetworks`', and upstream a 3-tensor argument is a
  `MethodError` because `*` cannot take one. So the witness is not a changed answer but a new one:
  a call that throws without this package loaded returns a value with it.

  Closing this means either `AbstractNeuralNetworks` gaining the three-axis methods, or this
  package wrapping the two layer types in its own. Both are an API change, and neither is a
  by-product of a piracy pass — which is why these three stayed when the other nine went.

  `src/layers/resnet.jl:63` is also the one remaining method ambiguity, against
  `AbstractNeuralNetworks.Affine`. That pair is benign and has no witness at all: `Dense` is not a
  subtype of `Affine` and the two have no common instance, so no call can reach it. `test/aqua.jl`
  asserts both counts — 3 piracies and 1 ambiguity — so neither can drift, in either direction.

### B10 · `DataLoader(::EnsembleSolution{T, T1, Vector{ST}})` at `src/data_loader/data_loader.jl:325` has no test and appears unreachable through this package's current dependencies.

- **location:** `src/data_loader/data_loader.jl:325`
- **kind:** missing test
- **found:** 2026-09-18
- **evidence:**
  It dispatches
  on `ST <: Union{GeometricSolution{T, T1, TT, NamedTuple{(:t, :q, :v), TuT}},
  GeometricSolution{T, T1, TT, NamedTuple{(:t, :q, :q̇), TuT}}}` — a `GeometricSolution` whose
  `dataser` has exactly the two keys `:q` and `:v` (or `:q̇`) besides `:t`, with no `:p`.

  Checked against `GeometricEquations` 0.21.3 (this package's resolved version): every equation
  type's own `initialstate(equ, t, ics, params)` reconstructs its `ics` from its own fixed field
  set regardless of what is passed in, and no type pairs `:v`/`:q̇` without also carrying `:p` —
  `SODE`/`ODE` give `(:q,)` alone; `PODE`/`HODE` give `(:q, :p)`; `IODE`/`LODE` give
  `(:q, :p, :v)`; `IDAE`/`LDAE` give `(:q, :p, :v, :λ, :μ)`. Verified directly for `SODE`:
  `initialstate(equ::SODE, t, ics, params) = (q = _statevariable(ics.q, periodicity(equ)),)`
  discards everything but `.q` even when `ics` already has a `:v` key. So no
  `EquationProblem`/`EnsembleProblem` built from any equation type this package depends on can
  produce a two-key `dataser` — passing a NamedTuple with the right keys through the public
  constructor does not help, because the equation-specific `initialstate` method throws it away.

  `GeometricSolution` and `EnsembleSolution` each define exactly one inner constructor (taking a
  `GeometricProblem`/`EnsembleProblem`), so Julia generates no default all-fields constructor for
  either, and there is no supported way to build one directly. The two low-level bypasses tried —
  `ccall(:jl_new_struct, ...)` on the (mutable) `GeometricSolution`, and
  `ccall(:jl_new_struct_uninit, ...)` followed by `setfield!` on each field — both crashed the
  Julia process with a segmentation fault (the first immediately; the second on a later, unrelated
  allocation, consistent with GC scanning a partially-initialized object), which is why neither is
  a technique this package's test suite should rely on.

  The fix in `src/data_loader/data_loader.jl:339` (`zeros(T, ...)` instead of untyped `zeros(...)`)
  is correct by inspection — it matches the file's three sibling constructors character for
  character — but is untested. Closing this means either GeometricEquations gaining an equation
  type whose `initialstate` returns exactly `(:q, :v)` or `(:q, :q̇)`, or deciding the method is
  dead code and removing it (Part E of the audit, not this one).

### B11 · `PositionalEncoding` is CPU-only, and the failure is a compile error rather than a slowdown.

- **location:** —
- **kind:** defect
- **found:** 2026-09-18
- **evidence:**
  `positional_encoding` builds its matrix with `Matrix{T}(undef, …)`, so the layer's
  `x .+ P` adds a host array to whatever it is given. Every other layer allocates through the
  backend — `KernelAbstractions.allocate`, or `similar(x, …)` in the forward pass. Broadcasting a
  host `Matrix` against a device array does not fall back to the CPU; it fails to compile, because
  the host array cannot be read from a kernel.

  **No test can catch this, because the suite has no GPU test.** That is what makes it worth an
  entry rather than a docstring alone: a green matrix says nothing about it. Both docstrings — the
  layer's and the `positional_encoding` keyword of [`Transformer`](@ref) — carry the warning.

  Closing it means giving the builder the backend, so that it allocates through
  `KernelAbstractions.allocate` and fills with a kernel instead of a loop, and it cannot be verified
  here without a GPU job to run it under.

## C. Follow-ups and cleanups

### C1 · The parameter-tree traversal still belongs upstream.

- **location:** —
- **kind:** upstream
- **found:** 2026-08-17
- **evidence:**
  `_make_optimizer_cache`,
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

### C2 · Two `isa` branches remain in `_leaf_optim_step!`

- **location:** —
- **kind:** defect
- **found:** 2026-08-16
- **evidence:**
  (for `AdamState`/`MomentumState`).
  Measurement showed the traversal is not implicated in the compile-time problem, so this is tidying,
  and it disappears entirely if C1 lands first.

### C6 · Three generated MNIST PDFs are in this branch's history

- **location:** `docs/src`
- **kind:** defect
- **found:** 2026-08-16
- **evidence:**
  for three commits, from a
  `git add docs/src` that swept them in. They are untracked again and `.gitignore` now covers the
  pattern, so the working tree and the net diff are clean, but the blobs are still reachable.
  Rewriting the branch would remove them.

### C7 · `SymbolicPullback(::HamiltonianArchitecture)` duplicates the upstream constructor.

- **location:** —
- **kind:** upstream
- **found:** 2026-08-16
- **evidence:**
  It has
  to, because `SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` derives the dimension of the
  loss's target from `output_dimension(nn.model)`, and for an HNN that is the scalar Hamiltonian
  rather than the vector field the loss compares against (see *Fixed*). Reproducing the constructor
  means reaching into three names that SymbolicNeuralNetworks does not export —
  `symbolic_parameter_gradient`, `ParameterGradient`, and the two-argument `SymbolicPullback` inner
  constructor — so an upstream refactor breaks GML silently at the type level. A keyword on the
  upstream constructor, or a `NetworkLoss` interface that states its own target dimension, would put
  this method back to one line.

  (**C9** is closed by this release and its entry is gone: `legacy/hnn/` and `legacy/mtk/` are
  deleted, and with them all 28 of the `include` sites it counted. The `data.jl` decision it was
  waiting on is moot — nothing needs that file any more. See *Removed (breaking)* above. The
  number is left vacant rather than reused.)

  (**C12** and **C13** are closed by this release and their entries are gone. C12 resolved to a
  design answer rather than a repair — approximate end-to-end symplecticity is the intent, and the
  verification script now states it. C13 resolved against the expectation its own entry recorded:
  `DEFAULT_LNN_NRUNS` is removed, and `∇L`, `∇∇L` and `∇q̇∇q̇L` are kept as the `Zygote` reference
  for what `LNNLoss` computes symbolically. Both are under *Changed*. The numbers are left vacant
  rather than reused.)

### C14 · `evaluate_vf_and_compute_∇Ψ` evaluates the decoder twice.

- **location:** `src/reduced_system/reduced_system.jl`
- **kind:** defect
- **found:** 2026-09-18
- **evidence:**
  `src/reduced_system/reduced_system.jl` calls `decoder((q = q̃, p = p̃))` for the value, then
  `ForwardDiff.jacobian(qp -> decoder(qp), vcat(q̃, p̃))` evaluates it again for the derivative. The
  comment above the function blamed "a problem with nested derivatives in ForwardDiff" and named no
  version, no issue and no reproducer; that claim is not reproduced here, and the comment now points
  at this entry instead of asserting it. What is certain is the double evaluation, which is visible
  in the two calls.

  Closing it means computing the value and the Jacobian in one pass — `DiffResults` is the tool —
  across the shape change from the `(q, p)` `NamedTuple` the vector fields are splatted from to the
  flat vector the Jacobian is taken against. `test/reduced_order_modeling/reduced_system.jl`
  exercises the function, so the change is checkable. It was left out of the audit's Part C
  deliberately: it is a restructuring with its own verification, not the comment repair that part
  was scoped to.

### C15 · The `kernel_ad_routines` buffers are still zeroed where allocating would do.

- **location:** `src/kernels/kernel_ad_routines/`
- **kind:** defect
- **found:** 2026-09-18
- **evidence:**
  `src/kernels/kernel_ad_routines/` allocates `dA = zero(A)`, `dS = KernelAbstractions.zeros(…)` and
  the same shape in `tensor_mat_mul.jl`, `tensor_mat_skew_sym_assign.jl` and `vec_tensor_mul.jl`.
  Their kernels sum into a local and assign the buffer element once, over an `ndrange` equal to the
  buffer's size, so no element is read before it is written — the same argument that let the forward
  wrappers move to `allocate` and `similar` under *Changed*.

  This part was scoped to the forward path, so they were not changed with it. The backward pass is
  where training spends its time, so the gain should be larger here than the forward figures, which
  is also why it wants its own before-and-after measurement rather than being folded into that pass.

### C16 · Four of `ExplicitImports`' seven checks are switched off in `test/aqua.jl`.

- **location:** `test/aqua.jl`
- **kind:** upstream
- **found:** 2026-09-18
- **evidence:**
  The counts, at
  the commit that added the gate: **37** names arriving through a bare `using` across thirteen
  packages;
  **11** explicit imports of names upstream does not mark `public`; **2** qualified accesses through a
  non-owner module, `GeometricOptimizers.Gradient` and `GeometricOptimizers.direction`, both owned by
  `SimpleSolvers`; and **22** qualified accesses to non-public names.

  They are four separate jobs, not one. Naming the 37 is a rewrite of the module header with a
  judgement per name. The 11 and the 22 are the same class and most of them are not this package's to
  fix — `Architecture`, `AbstractExplicitLayer`, `add!`, `_compute_loss`, `assign_columns` and
  `description` are all used deliberately, and the resolution is upstream declaring them `public`.
  Each check's reason is written above the `@testset` rather than here alone, because switching one
  back on turns the suite red on the spot.

### C17 · `accuracy` cannot be reached through any public `DataLoader` constructor.

- **location:** `src/data_loader/data_loader.jl:487`
- **kind:** defect
- **found:** 2026-09-18
- **evidence:**
  Its signature is
  `DataLoader{T, AT <: AbstractArray{T}, BT <: AbstractArray{T1}}` with `T1 <: Integer`
  (`src/data_loader/data_loader.jl:487`), and every `DataLoader(input, output)` method requires the
  two arrays to share one element type. A classifier's data are the case where they do not: real
  inputs, integer one-hot targets. So the only way to build an argument for this function is to name
  the parametric type, which is what `test/data_loader/accuracy.jl` does.

  Closing it is a `DataLoader(input::AbstractArray{T, 3}, output::AbstractArray{T1, 3})` constructor
  with the two element types free. That is an API addition rather than a cleanup, which is why the
  test records the gap instead of the same change closing it.

### C18 · `src/kernels/exponentials/tensor_exponential.jl` no longer defines `tensor_exponential`; it holds the identity tensor and its `rrule`, used by `tensor_cayley.jl` and `cpu_inverse.jl`.

- **location:** `src/kernels/exponentials/tensor_exponential.jl`
- **kind:** defect
- **found:** 2026-09-19
- **evidence:**
  The file name, and the `exponentials/` directory around it, are wider than what they hold. A
  comment in the file itself cites "issue C18" by number, so this entry exists to keep that reference
  valid. Renaming both the file and the directory is the fix.

### C19 · `_make_block_for_initialization` has two methods

- **location:** `src/architectures/symplectic_transformer.jl`
- **kind:** dead code
- **found:** 2026-09-19
- **evidence:**
  — `src/architectures/symplectic_transformer.jl`
  and `src/architectures/linear_symplectic_transformer.jl` — that differ only in which fields they read
  from their argument: one reads `arch.transformer_dim` and `arch.sympnet_activation`, the other reads
  `arch.dim` and `arch.activation`. The bodies are otherwise identical. This stood as a `TODO` in the
  source; a comment must stand on its own, so it is recorded here instead. Closing it means agreeing
  on one set of field names across the two architectures, or passing the dimension and the activation
  in rather than reading them off `arch`.

### C20 · There is no Knet.jl example for the Hamiltonian neural network.

- **location:** `TODO.md`
- **kind:** docs
- **found:** 2026-09-19
- **evidence:**
  The deleted `TODO.md` asked
  for one. Recorded rather than dropped; nothing depends on it.

### C21 · "An untrained autoencoder reduces worse than PSD" is not a property, and the suite no longer asserts it under an explicit integrator.

- **location:** `test/reduced_order_modeling/reduced_system.jl`
- **kind:** missing test
- **found:** 2026-09-21
- **evidence:**
  `test/reduced_order_modeling/reduced_system.jl` asserted both
  `projection_error(rs1) < projection_error(rs2)` and
  `reduction_error(rs1) < reduction_error(rs2)` for `ImplicitMidpoint()` and for
  `ExplicitMidpoint()`, with `rs1` a `PSDArch` and `rs2` an untrained 20-encoder-layer
  `SymplecticAutoencoder` over a softplus.

  The second one is a coin flip. `projection_error` compares the two autoencoders on the *full*
  solution, but `reduction_error` integrates the *reduced* system, and an untrained network's
  reduced vector field is one an explicit method diverges on — `reduction_error(rs2)` then comes
  back `NaN`, and `NaN` is not ordered against anything. Measured over seeds 1 to 12, identically
  at `GeometricOptimizers` 0.7.0 and 0.8.0:

  | assertion | `ImplicitMidpoint` | `ExplicitMidpoint` |
  |:--|:--|:--|
  | `projection_error(rs1) < projection_error(rs2)` | 12 / 12 | 11 / 12 |
  | `reduction_error(rs1) < reduction_error(rs2)` | 12 / 12 | **4 / 12**, with 4 `NaN` and 4 in the opposite order |

  The committed `Random.seed!(123)` is not among those twelve; it is a thirteenth draw, and it
  happened to be one that passed. The `GeometricOptimizers` bump in this release changes the
  manifold layers' orthonormalization from Householder `qr!` to CholeskyQR2, so that seed now
  draws a different network, and at 0.8.0 `reduction_error(rs2)` under `ExplicitMidpoint` is
  `NaN`. The bump exposed this; it did not cause it, and the table above is the same at 0.7.0.

  The explicit-integrator case now runs the projection-error assertion only, through a
  `compare_reduction_error` keyword, and the measurement is written above the function. **The seed
  was not changed**, because tuning it would have restored a green suite without restoring a true
  claim. Closing this entry means deciding what the reduction error of an untrained network is
  supposed to be worth testing at all — training the two networks first would make the comparison
  mean something, at a cost the suite does not currently pay.

## D. Unverified

Not defects — claims this release makes that nothing has actually checked yet.

### D4 · The upstream fix was measured on one optimizer.

- **location:** —
- **kind:** not verified
- **found:** 2026-08-16
- **evidence:**
  The compile-time figures come from the
  `Adam` path. The quasi-Newton and Newton caches and states were widened on the strength of their
  *inferred types* — a sound argument, but not a measurement. Catalogued upstream as GeometricOptimizers C15.
