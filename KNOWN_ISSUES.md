# Known issues

What is known to be wrong in GeometricMachineLearning and is not fixed.

## B. Known defects

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

### C6 · Three generated MNIST PDFs are in this branch's history

- **location:** `docs/src`
- **kind:** defect
- **found:** 2026-08-16
- **evidence:**
  for three commits, from a
  `git add docs/src` that swept them in. They are untracked again and `.gitignore` now covers the
  pattern, so the working tree and the net diff are clean, but the blobs are still reachable.
  Rewriting the branch would remove them.

### C7 · `SymbolicPullback(::HamiltonianArchitecture)` reaches one private SymbolicNeuralNetworks name.

- **location:** `src/pullbacks/symbolic_hnn_pullback.jl`
- **kind:** upstream
- **found:** 2026-08-16
- **evidence:**
  `SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` cannot be used, because it sizes the symbolic
  target of the loss with `output_dimension(nn.model)`, which for a Hamiltonian network is `1` — the
  scalar Hamiltonian — while `HNNLoss` compares against the Hamiltonian *vector field*, which has the
  dimension of the network's input. GML's construction therefore reaches
  `SymbolicNeuralNetworks.symbolic_parameter_gradient`, which no exported name replaces:
  `flat_parameter_gradient` is flat, so it wants an `unflatten` per step, and `ParameterGradient` is
  not used. The pullback is the upstream type `SymbolicPullback(loss, fun)`, with a call method of
  this package for `SymbolicPullback{<:HNNLoss, <:_HNNParameterGradient}` that is more specific
  than the upstream call in every argument. That private name is the one an upstream refactor
  breaks.

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
  flat vector the Jacobian is taken against. `test/reduced_system/reduced_system.jl`
  exercises the function, so the change is checkable. It was left out of the audit's Part C
  deliberately: it is a restructuring with its own verification, not the comment repair that part
  was scoped to.

### C16 · One of `ExplicitImports`' seven checks is switched off in `test/quality/aqua.jl`, and two run with a list of names.

- **location:** `test/quality/aqua.jl`
- **kind:** upstream
- **found:** 2026-09-18
- **evidence:**
  `no_implicit_imports` is off: **37** names arrive through a bare `using` across thirteen packages,
  and naming them is a rewrite of the module header with a judgement per name.
  `all_explicit_imports_are_public` and `all_qualified_accesses_are_public` run with a list of the
  names this package reaches in other packages that those packages do not declare `public`: **10**
  explicit imports and **9** qualified accesses, none of them in `GeometricOptimizers`.
  `Architecture`, `AbstractExplicitLayer`, `_compute_loss`, `dim` and the rest are used
  deliberately, and the resolution is upstream declaring them `public`. Each check's reason is
  written beside it in the test file.

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

- **location:** `test/reduced_system/reduced_system.jl`
- **kind:** missing test
- **found:** 2026-09-21
- **evidence:**
  `test/reduced_system/reduced_system.jl` asserted both
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

## Found late

### K1 · Two source comments still send the reader to the Open Issues section of `CHANGELOG.md`, which no longer exists.

- **location:** `src/reduced_system/reduced_system.jl:109`
- **kind:** found late
- **found:** 2026-09-26
- **evidence:**
  ```
  $ grep -rnE 'Open Issues|issue C1[48]|\*C21\*|\*B7\*' src test
  src/kernels/exponentials/tensor_exponential.jl:2:# name is wider than what it holds; renaming it, and `exponentials/` with it, is issue C18.
  src/reduced_system/reduced_system.jl:109:# issue C14 in `CHANGELOG.md`.
  test/reduced_system/reduced_system.jl:44:# for four of twelve, with four `NaN` and four in the opposite order. That is *C21* in
  ```

  The IDs are still valid; the comments at `src/reduced_system/reduced_system.jl:109` and
  `test/reduced_system/reduced_system.jl:44–45` name `CHANGELOG.md` rather than `KNOWN_ISSUES.md`.
  The fix is a follow-up change to those comments.

### K11 · A comment in `adam_with_learning_rate_decay.jl` reports a past experiment.

- **location:** `test/optimizers/adam_with_learning_rate_decay.jl`
- **kind:** found late
- **found:** 2026-10-12, by the critics of part M1a; the comment is older than that part
- **evidence:**
  The comment above `train_manifold_network` says "from a seeded start the loss fell by under 2%
  on 1.13 and *rose* on 1.12". The fix states only the present constraint: a 32-epoch budget
  drives the learning rate to `η₂` almost at once, and 128 epochs train.

### K15 · `HNNLoss` and `SymbolicPullback(arch)` accept `(q, p)` data in their signatures and fail on it inside the generated vector field.

- **location:** `src/loss/hnn_loss.jl:33`
- **kind:** found late
- **found:** #344, review round 2 (2026-10-14); the `HNNLoss` signatures are older than that pull request
- **evidence:**
  `HNNLoss`'s two functors (`src/loss/hnn_loss.jl:31-41`) and the call method of
  `SymbolicPullback(arch)` (`src/pullbacks/symbolic_hnn_pullback.jl:38`) type the data as
  `QPTOAT`, which admits a `(q = …, p = …)` `NamedTuple`. The loss hands that data to the generated
  `hvf`, which calls `ndims` on it. With `(q = randn(1, 5), p = randn(1, 5))` as input and output,
  the symbolic pullback, `HNNLoss` and `ZygotePullback(HNNLoss(arch))` all raise:

  ```
  MethodError: no method matching ndims(::@NamedTuple{q::Matrix{Float64}, p::Matrix{Float64}})
  ```

  Narrowing the pullback's method alone changes nothing: the data then reaches the upstream
  `SymbolicPullback` call, which also accepts a `NamedTuple` and fails in the same place. The fix
  belongs with the data signatures of the losses as a whole: either each loss accepts only the data
  its vector field can take, or `(q, p)` data is converted before it reaches the field.

### K16 · `scripts/utilities/inverse_generator.jl` reproduces the wrappers of the committed inverse files byte for byte, but not their kernels.

- **location:** `scripts/utilities/inverse_generator.jl:1`
- **kind:** found late
- **found:** 2026-10-15, by the builder of part M3; the same output comes from the generator of `origin/main` at `2a63f26a`
- **evidence:**
  Formatted with `style = "sciml"`, the generator writes the kernel of each of the four
  `inverse_NxN.jl` files, the part before `function tensor_inverseN(`, with the factors of some
  products in another order than the committed file, from line 6 or 7 on:

  ```
  <                 t2 = (*)((*)(t1, (getindex)(A, 2, 1, k)), (getindex)(A, 1, 2, k))
  ---
  >                 t2 = (*)((*)(t1, (getindex)(A, 1, 2, k)), (getindex)(A, 2, 1, k))
  ```

  The order depends on the Julia version, not on the package versions. With Symbolics 7.39.2,
  SymbolicUtils 4.46.6 and JuliaFormatter 2.14.0 fixed, Julia 1.11.9, 1.12.6, 1.12.7 and 1.13.1
  each give a different order, and none of them reproduces all four committed kernels; 1.12.6
  reproduces the 2×2 one. On Julia 1.13.1, SymbolicUtils 4.46.6 and 4.51.0 give identical output,
  and two runs on one Julia version give identical output. The Julia build that made the committed
  kernels on 2026-09-14 is no longer installed, so that run cannot be repeated.

  The rest of each file, from `function tensor_inverseN(` to the end, is byte for byte what the
  generator's `wrapper_source(n)` writes, and the script's header claims no more than that.

### K17 · `assign_q_and_p` and `PoissonTensor * v` drop the rows from `2N + 1` on without an error.

- **location:** `src/kernels/assign_q_and_p.jl`
- **kind:** defect
- **found:** #345, review of 2026-10-15; the behaviour is the same on `origin/main` at `2a63f26a`
- **evidence:**
  `assign_q_and_p(x, N)` takes `x[1:N, …]` for `q` and the next `N` rows for `p`, and checks nothing
  about `size(x, 1)`. An input with `size(x, 1) > 2N` loses its trailing rows silently, and
  `PoissonTensor * v` reaches the same method.

  Closing it means an error for `size(x, 1) != 2N`. That changes the public behaviour of
  `assign_q_and_p` and of `PoissonTensor * v`, so it is a change of its own and not a cleanup.

### K19 · In `Float32` the Cayley kernel returns non-finite slices for a skew `C` of norm 1e8, where no method is accurate.

- **location:** `src/kernels/cayley.jl`
- **kind:** found late
- **found:** 2026-10-17, by the critics of part M4
- **evidence:**
  On 200 skew `Float32` slices of `N = 5` whose largest `‖Cₖ‖₂` is 6.55e8, `tensor_cayley` returns
  non-finite values in 23 of them and raises no error. The scale is far outside check 3's `{1, 1e2}`,
  and no method is accurate there: `tensor_cayley5`, the closed form that stays, is non-finite in 197
  of the same 200 slices, and the dense solve `(I + Cₖ) \ (I − Cₖ)` is finite on all 200 but has an
  orthogonality error `‖ΛₖᵀΛₖ − I‖∞` of 7.3e4 at worst, which is no answer at that scale. What leaves
  the `Float32` range is the elimination's own intermediate values rather than a small pivot: on the
  two of those slices checked, the same elimination in `Float64`, whose range holds them, reaches a
  largest intermediate entry of 4.6e183 and 2.5e207, against a smallest pivot of 1.2e8 and 2.6e8, and
  `Float32`'s largest finite value is 3.4e38. The kernel's error on its 177 finite slices is at most
  284.

  ```julia
  R = 1.0f8 * randn(Float32, 5, 5, 200)
  C = R - permutedims(R, (2, 1, 3))
  nonfinite(L) = count(k -> !all(isfinite, L[:, :, k]), 1:200)
  nonfinite(tensor_cayley(C))    # 23 of 200
  nonfinite(tensor_cayley5(C))   # 197 of 200
  ```

## Upstream

### K2 · Revise prints EMFILE errors in the test log

- **location:** `test/quality/jet.jl:3`
- **kind:** upstream
- **found:** 2026-10-01
- **evidence:**
  JET 0.12 loads Revise, and Revise's file watcher runs out of file handles. On Julia 1.13.1,
  the full-run log of `run-tests.jl <repository> full` holds 5 blocks
  `UNHANDLED TASK ERROR: IOError: FolderMonitor: too many open files (EMFILE)` when the suite
  includes `test/quality/jet.jl`, and 0 in the same run at ea0c397, which does not include it.
  The blocks are not test failures: both runs are green, and every testset of ea0c397 has the
  same count in both.

### K4 · Metal.jl 1.11.1 needs a Metal device to precompile, so a sandboxed run with no Metal cache fails before the tests

- **location:** `test/Project.toml:16`
- **kind:** upstream
- **found:** 2026-10-02
- **evidence:**
  Metal.jl 1.11.1's precompile workload calls `mtlfunction(identity, Tuple{Nothing})`
  (`src/precompile.jl:18`), which calls `device()`. Inside the macOS sandbox the device list is
  empty. On Julia 1.11.9, where no precompiled Metal cache exists for these flags, a sandboxed
  `run-tests.jl <repository> full` stops before the first test:

  ```
  ERROR: LoadError: BoundsError: attempt to access 0-element Vector{Metal.MTL.MTLDevice} at index [1]
      @ ~/.julia/packages/Metal/oQt6k/lib/mtl/device.jl:29 [inlined]
      @ ~/.julia/packages/Metal/oQt6k/src/precompile.jl:18 [inlined]
  in expression starting at ~/.julia/packages/Metal/oQt6k/src/precompile.jl:3
  ```

  On Julia 1.13.1 the same run is green, because a Metal cache that was precompiled outside the
  sandbox (`compiled/v1.13/Metal/` in the depot) matches the flags of `Pkg.test()`. The test file
  itself records a skip where `Metal.functional()` is false; only the precompile fails. Closing this
  is a Metal.jl change: a workload that does not run where no device exists.

## Documentation

### K5 · The `PositionalEncoding` docstring names `test/metal/`, which does not exist

- **location:** `src/layers/positional_encoding.jl:75`
- **kind:** docs
- **found:** 2026-10-02
- **evidence:**
  The docstring says "the GPU tests in `test/metal/` do not cover this layer". The Metal tests
  are in `test/devices/metal.jl`, and `origin/main` at 97bf0d8 has no `test/metal/` directory.
  The fix is a change of that path under `src/`.

### K18 · The tensor-sampling formula in the `DataLoader` manual is one short

- **location:** `docs/src/data_loader/data_loader.md:143`
- **kind:** docs
- **found:** 2026-10-16
- **evidence:**
  Item 1 of *Sampling from a tensor* gives the start positions as
  `shuffle(1:(input_time_steps − seq_length − prediction_window))`. For the time-series loader
  without an output shown on that page, `number_of_start_positions` takes
  `n = input_time_steps − seq_length − prediction_window + 1`, and `number_of_batches` counts
  `cld(n · n_params, batch_size)` minibatches (`src/data_loader/batch.jl`). For the example's
  five time steps and `Batch(4, 2, 0)`, the formula gives three start positions per parameter
  and the code gives four. The correction is to add `+ 1` to the formula's upper bound.

## The test suite

### K3 · `test/quality/jet.jl` does not see an unstable kernel-launch argument, nor two barriers in the Poisson-tensor kernel

- **location:** `test/quality/jet.jl`
- **kind:** missing test
- **found:** 2026-10-01
- **evidence:**
  Julia 1.13.1, JET 0.12.2, `mutate.jl <repository> … quality/jet.jl`. Three mutants that put
  `Base.inferencebarrier` on an argument of a kernel launch SURVIVED (142 pass, 3 broken):
  `kernel!(c, Base.inferencebarrier(a), b, ndrange = size(c))` in
  `src/kernels/tensor_tensor_mul.jl`; `augment_zeros!(zero_tensor,
  Base.inferencebarrier(output_diff), seq_length, …)` in `src/data_loader/tensor_assign.jl`; and
  `skew_mat_mul_k!(C, S, Base.inferencebarrier(B), n, …)` in `src/kernels/mat_tensor_mul.jl`. At
  `target_modules = (GeometricMachineLearning,)` a barrier on a launch argument gives 0 reports,
  and the same barrier in an `@assert` of the same launcher gives 3: the kernel functor takes
  `args...`, so the dispatch sits in KernelAbstractions' frames, which the filter drops. In
  `assign_ones_for_poisson_tensor_kernel!` (`src/arrays/poisson_tensor.jl:95`) two mutants
  SURVIVED: a barrier on `n` in `map_index_for_poisson_tensor(i, n)`, and a barrier on the written
  array `J`; a barrier on the stored value is CAUGHT. JET never reports a dynamic dispatch in the
  kernel's own frame. A barrier on a value inside each kernel body is CAUGHT.

### K10 · The two training tests of `adam_with_learning_rate_decay.jl` run in `Float64` only.

- **location:** `test/optimizers/adam_with_learning_rate_decay.jl`
- **kind:** missing test
- **found:** 2026-10-12, by the critics of part M1a; the precision is older than that part
- **evidence:**
  `train_network` and `train_manifold_network` build their `DataLoader` from
  `sin.(0:0.1:2π)`, a `Vector{Float64}`, and take the network's element type from `eltype(dl)`.
  Neither function takes an element type, so neither runs over `REAL_ELTYPES`. The third testset
  of the file, `schedule_starts_at_one`, loops over `REAL_ELTYPES`. The fix builds `dl` per `T`
  and loops both functions over `REAL_ELTYPES`.

### K14 · `train!` infers its history as an abstract `Vector`, although the run-time history is `Vector{T}`.

- **location:** `NeuralNetworkParameters ext/ZygoteRulesExt.jl:25`
- **kind:** upstream
- **found:** 2026-10-14, part M1b; M1a verifier finding #3 (2026-10-13)
- **evidence:**
  In a cold process, at the first call, `@inferred train!(nn, opt, dl, Batch(8), 2; loss =
  FeedForwardLoss(), show_progress = false)` on `Chain(Dense(2, 2, identity))` fails in both
  precisions:

  ```
  PRECISION  = Float64
  threw      = return type Vector{Float64} does not match inferred return type Vector
  PRECISION  = Float32
  threw      = return type Vector{Float32} does not match inferred return type Vector
  ```

  `DataLoader`'s minibatch is a pair of 3-tensors, `Tuple{Array{Float32,3}, Array{Float32,3}}`, and
  `ZygotePullback`'s value infers on it as follows:

  ```
  Float32 minibatch data => Tuple{Array{Float32, 3}, Array{Float32, 3}}
  Float32 Zygote.pullback value => Any
  ```

  The same call on a pair of matrices infers `Float32`, and the forward loss and `model(x, ps)`
  infer concretely on the 3-tensors as well. The first `Any` in `@code_warntype` of
  `Zygote.pullback(f, ::NetworkParameters)` on that minibatch is the one inside Zygote's own
  pullback: the value first becomes `Any` at `Zygote._pullback(Context(), f, ps)`, which is `Any` on
  the 3-tensors and `Float32` on the matrices, and NeuralNetworkParameters reaches it through
  `invoke(ZygoteRules.pullback, Tuple{Any, Vararg{Any}}, f, ps)` at `ext/ZygoteRulesExt.jl:25`,
  whose generic method is Zygote's own. Its 3-tensor rows, both precisions:

  ```
  Float32 3-tensor forward f(::NetworkParameters)                                    => Float32
  Float32 3-tensor ZygotePullback value                                              => Any
  Float32 3-tensor Zygote.pullback(f, ::NetworkParameters) value                     => Any
  Float32 3-tensor Zygote._pullback(Context(), f, ::NetworkParameters) value         => Any
  Float64 3-tensor forward f(::NetworkParameters)                                    => Float64
  Float64 3-tensor ZygotePullback value                                              => Any
  Float64 3-tensor Zygote.pullback(f, ::NetworkParameters) value                     => Any
  Float64 3-tensor Zygote._pullback(Context(), f, ::NetworkParameters) value         => Any
  ```

  `Zygote.pullback(z -> sum(model(z, ps)), x)` on the 3-tensor alone is `Any` too, so the widening
  sits in Zygote's adjoint of the layer rather than in the loss or the wrapper. `train!` accumulates
  its history over those values, so its inferred return type is `Vector`.

  The call's run-time result is `Vector{T}`, and every test asserts `eltype(history) === T`; what is
  lost is the compile-time type, not the behaviour. `ZygotePullback`'s own call carries no return
  annotation. A `Base.inferencebarrier` on the loss value hides the type instead of restoring it,
  and the same widening reaches a caller that builds its own loop on `ZygotePullback` without
  `train!`. The fix belongs to Zygote, whose adjoint of `AbstractNeuralNetworks`' `Dense` on a
  3-tensor gives the `Any`.
