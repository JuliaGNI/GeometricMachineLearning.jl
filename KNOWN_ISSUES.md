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

### C21 · "An untrained autoencoder projects worse than PSD" is not a property, and the suite asserts it on one draw.

- **location:** `test/reduced_system/reduced_system.jl`
- **kind:** missing test
- **found:** 2026-09-21
- **evidence:**
  `test_reduced_vector_fields` asserts `projection_error(rs1) < projection_error(rs2)` under
  `ImplicitMidpoint()` and under `ExplicitMidpoint()`, in `Float32` and in `Float64`, with `rs1` a
  `PSDArch` and `rs2` an untrained 20-encoder-layer `SymplecticAutoencoder` over a softplus. The
  file seeds its stream once, with `Random.seed!(123)`, so the four assertions rest on the draws
  that seed gives in the order the file makes them. With the seed set before each call of
  `set_up_reduced_systems`, over seeds 1 to 12:

  | element type | `ImplicitMidpoint` | `ExplicitMidpoint` |
  |:--|:--|:--|
  | `Float64` | 12 / 12 | 11 / 12 |
  | `Float32` | 11 / 12 | 9 / 12 |

  The reduction errors are asserted finite under `ImplicitMidpoint()` and are not ordered: the
  reduced vector field of an untrained network is one an explicit method diverges on, and
  `reduction_error(rs2)` then comes back `NaN`, which is not ordered against anything. Closing this
  entry means deciding what of an untrained network is worth testing at all — training the two
  networks first would make the comparison mean something, at a cost the suite does not pay.

## Found late

### K17 · `assign_q_and_p` drops the rows from `2N + 1` on without an error.

- **location:** `src/kernels/assign_q_and_p.jl`
- **kind:** defect
- **found:** #345, review of 2026-10-15; the behaviour is the same on `origin/main` at `2a63f26a`
- **evidence:**
  `assign_q_and_p(x, N)` takes `x[1:N, …]` for `q` and the next `N` rows for `p`, and checks nothing
  about `size(x, 1)`. An input with `size(x, 1) > 2N` loses its trailing rows silently.

  Closing it means an error for `size(x, 1) != 2N`. That changes the public behaviour of
  `assign_q_and_p`, so it is a change of its own and not a cleanup.

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
  `ParametricDataLoader`'s `(q, p)` constructor has the same one-type restriction and widens with
  it.

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

### K28 · AbstractNeuralNetworks' loss norm of a `(q, p)` named tuple promotes `Float32` to `Float64`.

- **location:** AbstractNeuralNetworks 0.9.3, `src/losses.jl:73`
- **kind:** upstream
- **found:** 2026-10-25, by the critics of the parametric core
- **evidence:**
  `_norm(dx::NamedTuple) = sum(map(_norm, values(dx))) / √length(dx)` divides by the square root of
  an `Int`, which is a `Float64`. So `ParametricLoss` returns a `Float64` on `Float32` `(q, p)`
  data, and `train!` on a `Float32` `(q, p)` `ParametricDataLoader` returns a `Vector{Float64}`
  history. Array data keep `Float32`.

  ```julia
  import AbstractNeuralNetworks
  x = (q = rand(Float32, 2, 1, 3), p = rand(Float32, 2, 1, 3))
  typeof(AbstractNeuralNetworks._norm(x))    # Float64
  ```

### K30 · The gradient of AbstractNeuralNetworks' loss of a `(q, p)` named tuple infers as `Union{Nothing, NetworkParameters}`.

- **location:** AbstractNeuralNetworks 0.9.3, `src/losses.jl:86`
- **kind:** upstream
- **found:** #357, review of 2026-10-26; the same inference shows on `origin/main` at `a664f5c9`
- **evidence:**
  On Julia 1.13.2 and `a664f5c9`, `Base.return_types` of the pullback of
  `ps -> _compute_loss((q = ps.L1.W * q, p = ps.L1.b .* p), (q = yq, p = yp))` gives
  `Tuple{Union{Nothing, NetworkParameters}}` in `Float32` and `Float64`, and the same function
  with `vcat` in place of the named tuples gives the concrete `NetworkParameters{T, (:L1,), …}`. So
  `ZygotePullback(ParametricLoss())` on `(q, p)` data infers its gradient as that `Union`, for one
  sample and for a batch, and a plain sum of squares over the `(q, p)` output of a batch infers
  concretely. The run-time gradient is a `NetworkParameters`.

  ```julia
  using GeometricMachineLearning, Zygote
  import AbstractNeuralNetworks
  ps = NetworkParameters((L1 = (W = rand(1, 1), b = rand(1)),))
  q, p, yq, yp = rand(1), rand(1), rand(1), rand(1)
  v, pb = Zygote.pullback(ps -> AbstractNeuralNetworks._compute_loss(
      (q = ps.L1.W * q, p = ps.L1.b .* p), (q = yq, p = yp)), ps)
  Base.return_types(pb, (typeof(v),))    # Tuple{Union{Nothing, NetworkParameters}}
  ```

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

### K31 · `changebackend` cannot move a parameter set onto a `JLBackend`.

- **location:** `AbstractNeuralNetworks` `src/utils/changebackend.jl:3`
- **kind:** upstream
- **found:** 2026-10-26, part M7
- **evidence:**
  `changebackend(backend, x::AbstractArray)` allocates with `KernelAbstractions.allocate` and fills
  the copy through `KernelAbstractions.copyto!(backend, dest, src)`. `JLArrays` implements no
  `copyto!(::JLBackend, ::JLArray, ::Array)`:

  ```
  julia> KernelAbstractions.copyto!(JLBackend(), JLArray(zeros(Float32, 2)), zeros(Float32, 2))
  ERROR: MethodError: no method matching copyto!(::JLBackend, ::JLArray{Float32, 1}, ::Vector{Float32})
  ```

  so

  ```julia
  changebackend(JLBackend(), params(NeuralNetwork(Chain(Dense(4, 8, tanh)), CPU(), Float32)))
  ```

  raises a `MethodError` whose message names `copyto!` rather than `changebackend`. The device tests
  therefore move a parameter set leaf by leaf: `test/helpers/device.jl`'s `to_device` rebuilds each
  structured leaf around `JLArray` storage. A `MetalBackend` has no such gap —
  `changebackend(get_backend(MtlArray(zeros(Float32, 1))), params(nn))` returns the set with every
  leaf's storage on the device and its type kept, `StiefelManifold` and `NamedTuple` leaves included —
  so `test/devices/metal_check.jl` uses it. The fix is a `copyto!` method in `JLArrays`, or a device-array
  path through `Base.copyto!` in `AbstractNeuralNetworks`.

### K34 · A `StiefelManifold` weight of one column gives an empty skew block, and a Metal `rmul!` of an empty array raises.

- **location:** `GeometricOptimizers` `src/lie_algebras/stiefel_lie_algebra_horizontal.jl:178`
- **kind:** upstream
- **found:** 2026-10-26, part M7
- **evidence:**
  The horizontal lift of a Stiefel point of `N` rows and `n` columns carries a `SkewSymMatrix` of
  dimension `n`, whose storage holds `n(n - 1) / 2` entries — none at all where `n` is one. The
  optimizer's cache and state then hold empty storages for that leaf, and a Metal launch of an empty
  array divides by zero:

  ```
  julia> train!(device, opt, DataLoader(MtlArray(randn(Float32, 8, 12, 4)); autoencoder = true), Batch(6), 2)
  ERROR: DivideError: integer division error
    [5] mod at int.jl:327 [inlined]
    [6] #partition##2 at nditeration.jl:158 [inlined]
   [11] launch_config at MetalKernels.jl:91 [inlined]
   [12] (::KernelAbstractions.Kernel{…generic_rmul_kernel!…})
   [14] generic_rmul! at linalg.jl:981 [inlined]
   [15] rmul! at linalg.jl:985 [inlined]
   [23] _StorageStep at named_tuple_wrapper.jl:114 [inlined]
  ```

  for `SymplecticAutoencoder(8, 2)`, whose parameter set holds `StiefelManifold` weights of `(4, 1)`
  and `(2, 1)`. Walking the optimizer after the failure finds the empty storage in each of
  `cache.g`, `cache.δ`, `cache.Δg`, `cache.g̃`, `cache.m₁`, `cache.m₂` and `state.g`, `state.ḡ`,
  `state.m₁`, `state.m₂`:

  ```
  ZERO opt.cache.g.params[L5][weight](parent)[1] :: SkewSymMatrix{Float32, MtlVector{Float32, Metal.PrivateStorage}} (0,)
  ZERO opt.cache.g.params[L7][weight](parent)[1] :: SkewSymMatrix{Float32, MtlVector{Float32, Metal.PrivateStorage}} (0,)
  ```

  A CPU- or `JLArray`-backed launch over an empty range is a no-op, which is why
  `test/devices/jlarray_autoencoder.jl` trains this network and no test sees the defect. The fix is a
  `rmul!` that skips an empty array on the device side, or a horizontal lift that allocates no block
  for `n = 1`. Until then `test/devices/metal_check.jl` sweeps `SymplecticAutoencoder(8, 4)`, whose
  manifold weights have two and three columns.

### K33 · An epoch of `train!` on a Metal array advances the CPU random stream, so a two-epoch run draws its second epoch's minibatches differently from the host's.

- **location:** `src/data_loader/batch.jl:195`
- **kind:** not verified
- **found:** 2026-10-26, part M7
- **evidence:**
  `train!` draws its minibatches with `shuffle` from the task-local stream, and nothing else. From one
  seed, the host's epoch leaves that stream exactly where the two shuffles left it, and the device's
  epoch does not:

  ```
  after the seed:  0.5805097055341872
  host  one epoch: Float32[1.2209982]  rng next 0.5692702072139225
  device one epoch: Float32[1.2209982]  rng next 0.24892438522171267
  ```

  The draw is in the device path and not in the model: a single gather of a minibatch on the device
  moves the stream, and so does a single `optimization_step!`, while the same calls on the host move
  it not at all. Two epochs inside one `train!` therefore hand the device and the host different
  minibatches from the second epoch on, which reads as a disagreement of `1e-2` in the loss history.
  `test/devices/metal_check.jl` works around it by calling `train!` once per epoch with the stream reseeded
  before each. Which of `Metal`, `GPUArrays`, `GeometricOptimizers` or `Zygote` draws is not
  established.

### K35 · Neither of the two Metal process aborts of 2026-09-22 reproduces in 59 fresh processes per arm, and neither run was ever shown to abort on demand.

- **location:** `scripts/utilities/metal_abort_runs.jl`
- **kind:** not verified
- **found:** 2026-10-27
- **evidence:**
  A `LASympNet(4)` process died once from `MTLReleaseAssertionFailure` in
  `-[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]` while a second session `ex` read a Julia
  vector, and a `VolumePreservingFeedForward(4)` process once from
  `BUG IN CLIENT OF LIBMALLOC: memory corruption of free block` inside the host LLVM JIT with no
  second `ex`. Both ran `Float32` on `MetalBackend()`, Adam, and died in the first training call.
  Neither recurred in a fresh process or in 24 warm runs, and the cause is not known.

  `scripts/utilities/metal_abort_runs.jl`, at the commit `7fdc3f8` that the runs used, runs one
  counted run per OS process: it
  prints its pid, `MTL_DEBUG_LAYER`, `Threads.nthreads()`, the version and path of every package of
  the stack, `Metal.functional()` and the device's name before the first call; it takes a forward
  pass and a Zygote pullback on one network and then times the *first* training call of a new one
  with `@timed Base.invokelatest`, so that the compilation is inside the timer; and it attaches a
  `julia-*.ips` to a run only when the report's `"pid"` is the run's own, so that the sixteen reports
  of other Julia processes in `~/Library/Logs/DiagnosticReports` are not counted. `N = 59` per arm,
  fixed before the first run: at a rate of one abort in twenty training calls, 59 clean runs happen
  with probability at most `0.95^59 = 0.0485`.

  | arm | process | N | clean | aborts | void |
  |:--|:--|:--|:--|:--|:--|
  | `LASympNet(4)`, no concurrent evaluation | `julia --project=<scratch> --threads=16`, one per run | 59 | 59 | 0 | 0 |
  | `VolumePreservingFeedForward(4)`, no concurrent evaluation | the same | 59 | 59 | 0 | 0 |
  | `LASympNet(4)` with a concurrent evaluation | two `ex` calls in a private Kaimon session | 59 | — | — | — |

  Every counted run compiled inside the timer (`compile_time` 15.6 s to 25.3 s for `LASympNet(4)`,
  11.6 s to 29.1 s for `VolumePreservingFeedForward(4)`), reported a finite `Float32` loss history,
  showed `Metal API Validation Enabled`, and had a pid no other run used. The crash reporter wrote no
  new `julia-*.ips` at any point in the 118 runs.

  The stack of the runs is Julia 1.13.3, this package at `1d2faead`, `GeometricOptimizers` 0.9.1,
  Metal.jl 1.11.1, GPUCompiler 2.10.0, KernelAbstractions 0.9.45 and LLVM.jl 9.13.2. The stack of the
  2026-09-22 aborts was Julia 1.13.0, Metal.jl 1.11.1, GPUCompiler 2.8.1, KernelAbstractions 0.9.42
  and LLVM.jl 9.13.1: **the same Metal.jl**, whose `flush_batched_queues!` walks every active
  batched queue and flushes it with no lock, the caller's task's or another task's. Metal.jl 1.12.0
  is not selectable from this environment: `Pkg.status(; outdated = true)` prints
  `Metal v1.11.1 (<v1.12.0)` with no constraint beside it and `KernelAbstractions v0.9.45
  (<v0.10.3)` restricted by this package, `GeometricOptimizers`, `AbstractNeuralNetworks` and
  Metal.jl, and which of the two holds the other down is not established.

  The third arm is not counted here. Its concurrency is a second request to one session, which no
  script can make: `scripts/utilities/metal_abort_session.jl` holds its two halves, and
  `Tasks/Evidence/…/m8/concurrent-arm.md` gives the procedure and the shape of its log, which
  `metal_abort_runs.jl audit` counts by the same rule. Two of the three arms are clean at 59 of 59;
  the third is that run.

  A no-reproduction holds for the two arms counted, on that stack, and says nothing about a true rate
  below one in twenty, nor about the session arm.

## The test suite

### K3 · `test/quality/jet.jl` does not see an unstable kernel-launch argument

- **location:** `test/quality/jet.jl`
- **kind:** missing test
- **found:** 2026-10-01
- **evidence:**
  Julia 1.13.2, `mutate.jl <repository> <file> <from> <to> quality/jet.jl`, one mutant per run.
  Three mutants that put `Base.inferencebarrier` on an argument of a kernel launch SURVIVED, each
  with 50 pass, 0 fail: `kernel!(c, Base.inferencebarrier(a), b, ndrange = size(c))` in
  `src/kernels/tensor_tensor_mul.jl`; `tensor_mat_skew_sym_assign_k!(C,
  Base.inferencebarrier(Z), A, ndrange = size(C))` in
  `src/kernels/inverses/tensor_mat_skew_sym_assign.jl`; and `gather_kernel!(backend)(gathered,
  Base.inferencebarrier(data), indices, offset, ndrange = size(gathered))` in
  `src/data_loader/batch.jl`. At `target_modules = (GeometricMachineLearning,)` a barrier on a
  launch argument gives 0 reports, and the same barrier in an `@assert` of the same launcher is
  CAUGHT with 3 reports (47 pass, 3 fail): the kernel functor takes `args...`, so the dispatch sits
  in KernelAbstractions' frames, which the filter drops. A barrier on a value inside the kernel
  body of `tensor_tensor_mul_kernel!` is CAUGHT (47 pass, 3 fail).

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
