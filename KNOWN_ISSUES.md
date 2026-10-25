# Known issues

What is known to be wrong in GeometricMachineLearning and is not fixed.

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

### C16 · One of `ExplicitImports`' seven checks is switched off in `test/quality/aqua.jl`, and two run with a list of names.

- **location:** `test/quality/aqua.jl`
- **kind:** upstream
- **found:** 2026-09-18
- **evidence:**
  `no_implicit_imports` is off: **33** names arrive through a bare `using` across twelve packages,
  and naming them is a rewrite of the module header with a judgement per name.
  `all_explicit_imports_are_public` and `all_qualified_accesses_are_public` run with a list of the
  names this package reaches in other packages that those packages do not declare `public`: **9**
  explicit imports and **7** qualified accesses, none of them in `GeometricOptimizers`.
  `Architecture`, `AbstractExplicitLayer`, `_compute_loss`, `dim` and the rest are used
  deliberately, and the resolution is upstream declaring them `public`. Each check's reason is
  written beside it in the test file.

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
  and no method is accurate there: the dense solve `(I + Cₖ) \ (I − Cₖ)` is finite on all 200 but has an
  orthogonality error `‖ΛₖᵀΛₖ − I‖∞` of 7.3e4 at worst, which is no answer at that scale. The cause
  is cancellation in the last pivot, not overflow: in all 23 non-finite slices the last `Float32`
  pivot is exactly `0`. The same elimination in `Float64` gives a last pivot of magnitude 1.7 to 15
  on those slices, and no intermediate entry above 6.8e8. In `Float32` the rounding of the earlier
  steps, about `eps(Float32) · ‖Cₖ‖₂ ≈ 78`, is larger than that pivot; for odd `N` the skew `Cₖ` is
  singular, so the pivot of `I + Cₖ` is small at that scale. The kernel's error on its 177 finite
  slices is at most 284.

  ```julia
  using GeometricMachineLearning: tensor_cayley
  import Random
  Random.seed!(3)
  R = 1.0f8 * randn(Float32, 5, 5, 200)
  C = R - permutedims(R, (2, 1, 3))
  nonfinite(L) = count(k -> !all(isfinite, L[:, :, k]), 1:200)
  nonfinite(tensor_cayley(C))    # 23 of 200
  ```

### K20 · The `DataLoader((q = …, p = …))` constructors require one array type for `q` and `p`.

- **location:** `src/data_loader/data_loader.jl:216`
- **kind:** defect
- **found:** #353 (2026-10-23); the constructors are older than that pull request
- **evidence:**
  The three methods at `src/data_loader/data_loader.jl:216-245` take
  `NamedTuple{(:q, :p), Tuple{AT, AT}}` (and `Tuple{VT, VT}`), so `q` and `p` must have one
  array type, while the `TransformerLoss` method at `src/loss/losses.jl:92` and the
  `MultiHeadAttention` call take `NamedTupleOfArrays`, which admits two. A view for `q` and a
  `Matrix` for `p` raises:

  ```julia
  A = randn(2, 10); B = randn(2, 10)
  DataLoader((q = view(A, :, 1:10), p = B); suppress_info = true)
  # MethodError: no method matching DataLoader(::@NamedTuple{q::SubArray{Float64, 2, …}, p::Matrix{Float64}})
  ```

  The same call with `q = A` gives a `DataLoader` with `input_dim == 4`. The docstring at
  `src/data_loader/data_loader.jl:205` states the one-type signature. Closing it means widening the
  three signatures to `NamedTupleOfArrays{T, (:q, :p)}`, and the docstring with them.

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
