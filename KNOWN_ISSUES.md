# Known issues

What is known to be wrong in GeometricMachineLearning and is not fixed.

## C. Follow-ups and cleanups

### C6 · Three generated MNIST PDFs are in the history of `main`

- **location:** `docs/src/tutorials/mnist/*.pdf`
- **kind:** defect
- **found:** 2026-08-16
- **evidence:**
  `git log --all --diff-filter=A --name-only -- 'docs/src/tutorials/mnist/*.pdf'` names the three of
  them, swept in by a `git add docs/src`, and the same command with `--diff-filter=D` names the two
  commits that untracked them again: added in `8dc9b831` and `33882583`, deleted in `a69f4641` and
  `b0010737`. Eight commits of `origin/main` carry the three files in their tree, so the blobs are
  still reachable in the history of `main`. `8e3dbc29`, which moved the MNIST material to
  GMLDatasets, deleted the `.tex` sources, their `Makefile` and the `.gitignore` rule for their
  PDFs, so no build of this tree compiles them a second time. Rewriting the history would remove
  the blobs.

### C16 · One of `ExplicitImports`' seven checks is switched off in `test/quality/aqua.jl`, and three run with a list of names.

- **location:** `test/quality/aqua.jl`
- **kind:** upstream
- **found:** 2026-09-18
- **evidence:**
  `no_implicit_imports` is off: **33** names arrive through the ten bare `using` statements in the
  module header. Ten modules export them — the ten packages those statements name — and eleven own
  them: `ZygoteRules` owns one of the 33, which `Zygote` re-exports. Naming them all is a rewrite of
  the module header with a judgement per name.
  `all_explicit_imports_are_public` and `all_qualified_accesses_are_public` run with a list of the
  names this package reaches in other packages that those packages do not declare `public`: **7**
  explicit imports and **7** qualified accesses, none of them in `GeometricOptimizers`.
  `Architecture`, `AbstractExplicitLayer`, `_compute_loss`, `dim` and the rest are used
  deliberately, and the resolution is upstream declaring them `public`.
  `no_stale_explicit_imports` runs with `update!` on its ignore list: the import has no use in
  `src/`, and it keeps `GeometricMachineLearning.update!` the binding of the `GeometricOptimizers`
  verb. Each check's reason is written beside it in the test file.

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

### K40 · `HRedSys` carries a cache's buffers and its `pullback_mode` in fields that no code in `src/` reads.

- **location:** `src/reduced_system/reduced_system.jl:107`
- **kind:** dead code
- **found:** #370
- **evidence:**
  The constructor sets `pullback_mode`, `ξ_buf`, `v_intermediate`, `f_intermediate`, `jac_result`
  and `dual_cache`, and the type parameter `JR` carries the type of `jac_result`. `v_reduced` and
  `f_reduced` hold the `ReducedVFCache` and the mode in their closures, and every read of
  `.ξ_buf`, `.v_intermediate`, `.f_intermediate`, `.jac_result` and `.dual_cache` in `src/` is a read
  of a `ReducedVFCache`, never of an `HRedSys`. `test/reduced_system/reference_shift_and_vjp.jl`
  asserts the fields with `hasfield`. The fields stay because code outside this package may need
  them; removing them changes the type's parameters, its layout and those tests.

### K41 · `CrossModalLoss(nn, x_a, x_b)` with a mixed-component `NamedTuple` raises a `MethodError`, not the documented `ArgumentError`.

- **location:** `src/loss/multimodal_losses.jl:382`
- **kind:** defect
- **found:** #370
- **evidence:**
  The three `(nn, x_a, x_b)` methods (lines 382, 388 and 394) annotate `x_a` and `x_b` with
  `ArrayOrNamedTuple`, and AbstractNeuralNetworks defines that as
  `Union{AbstractArray{T}, NamedTupleOfArrays{T}}`, with one element type `T` for every component.
  A `NamedTuple` whose `q` and `p` differ in element type matches no `T`, so dispatch fails before
  `_check_eltype` (line 251) runs. In a fresh `julia --startup-file=no --project=.` process:

  ```julia
  T = typeof((q = zeros(Float32, 2, 3), p = zeros(Float64, 2, 3)))
  T <: AbstractNeuralNetworks.ArrayOrNamedTuple   # false
  ```

  The pair form `CrossModalLoss(nn, (x_a, x_b))` takes `Tuple{Any, Any}`, reaches
  `_check_eltype`, and is tested. The docstring says that another element type raises an
  `ArgumentError`. Closing this entry means widening the `x_a` and `x_b` annotations of the three
  methods, for example to `Union{AbstractArray, NamedTuple}`. That is a design call about what the
  method accepts, and the pull request does not make it.

### K46 · The dual-number caches of `ReducedVFCache` and `HRedSys` are fields of type `IdDict{DataType, Any}`, whose value type is `Any`.

- **location:** `src/reduced_system/reduced_system.jl:7`
- **kind:** defect
- **found:** 2026-11-09
- **evidence:**
  `julia --startup-file=no concrete-fields.jl <package>` (Julia 1.13, at `d9cebdef`) reports two
  of its fields:

  ```
  GeometricMachineLearning.HRedSys.dual_cache::IdDict{DataType, Any}
  GeometricMachineLearning.ReducedVFCache.dual_cache::IdDict{DataType, Any}
  ```

  `ReducedVFCache.dual_cache` (line 7) maps the element type `T` of a state that is not an
  `AbstractFloat`, such as a `ForwardDiff.Dual`, to a `ReducedVFCache(T, N2, n2)`, which
  `_get_vf_cache` builds on first use (lines 30–32). The read asserts `::ReducedVFCache{T}`, which
  leaves the parameter `JR` open, so the buffers infer as `Vector{T}` and `c.jac_result`, the
  first argument of `ForwardDiff.jacobian!` (line 263), does not infer. The `::Vector{T}` and
  `::Matrix{T}` assertions at lines 264 and 273 restore the types after that call. Of the two
  callers of `_get_vf_cache`, only `evaluate_vf_and_compute_∇Ψ` (line 257) reaches the
  dictionary: `evaluate_vf_and_vjp` (line 282) runs only for an `AbstractFloat` element type
  (lines 339 and 381), for which `_get_vf_cache` returns at line 23 or throws at line 25.
  `HRedSys.dual_cache` (line 146) holds the same dictionary (line 197), and no code in `src/`
  reads it. `test/reduced_system/reference_shift_and_vjp.jl` asserts its type at line 184 and
  reads it at line 196. A fix is a design decision: a dual cache whose element type is a type
  parameter that the caller fixes, or the dictionary with the full concrete type asserted at the
  read. The first changes the type that the test asserts. The second makes the read infer and
  still leaves the field flagged.

### K47 · The section "GPU Support" of the manual reports device results measured on versions that `[compat]` no longer admits.

- **location:** `docs/src/index.md:55`
- **kind:** docs
- **found:** #381
- **evidence:**
  The page says that what follows "was measured by hand, before the move to
  `KernelAbstractions.jl` 0.10, on an Apple M4 Max through `Metal.jl` 1.11.1, in `Float32`, … and
  against `GeometricOptimizers` 0.9.0". `test/Project.toml` takes
  `Metal = "1.12"` and `Project.toml` takes `GeometricOptimizers = "0.10"`, so no environment of
  this version holds either. The list of architectures that ran on the device is not measured on
  Metal 1.12 and GeometricOptimizers 0.10. Closing this entry means running that list again by hand
  on a Metal device and updating the paragraph.

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

- **location:** `ReducedComplexityModeling` `src/data_loader/data_loader.jl:220`
- **kind:** defect
- **found:** #353 (2026-10-23); the constructors are older than that pull request
- **evidence:**
  The three methods at `ReducedComplexityModeling` `src/data_loader/data_loader.jl:220-249`
  take `NamedTuple{(:q, :p), Tuple{AT, AT}}` (and `Tuple{VT, VT}`), so `q` and `p` must have one
  array type, while the `TransformerLoss` method at `src/loss/losses.jl:98` and the
  `MultiHeadAttention` call take `NamedTupleOfArrays`, which admits two. A view for `q` and a
  `Matrix` for `p` raises:

  ```julia
  A = randn(2, 10); B = randn(2, 10)
  DataLoader((q = view(A, :, 1:10), p = B); suppress_info = true)
  # MethodError: no method matching DataLoader(::@NamedTuple{q::SubArray{Float64, 2, …}, p::Matrix{Float64}})
  ```

  The same call with `q = A` gives a `DataLoader` with `input_dim == 4`. The docstring at
  `ReducedComplexityModeling` `src/data_loader/data_loader.jl:208` states the one-type signature.
  Closing it means widening the three signatures to `NamedTupleOfArrays{T, (:q, :p)}`, and the
  docstring with them. `ParametricDataLoader`'s `(q, p)` constructor has the same one-type
  restriction and widens with it.

### K37 · `metal_abort_runs.jl audit` throws when a run's crash report is gone, although its log records the signature.

- **location:** `scripts/utilities/metal_abort_runs.jl:293`
- **kind:** defect
- **found:** #365
- **evidence:**
  `read_run` calls `report_signature(report_path(path, index, file, reports))` for every `M8-IPS` line.
  `report_path` returns the copy beside the log when it exists and else `joinpath(reports, file)`, and
  `report_signature` is `read(path, String)` on it (`:101-102`), so a copy that is missing with the
  original deleted from `~/Library/Logs/DiagnosticReports` raises a `SystemError` and the audit of the
  arm stops. Each `M8-IPS` line already holds `signature=` (`:426-427`), which the audit could use when
  the file is absent. Read from the code; the failure is not reproduced.

### K38 · The session arm's reader stamps `R0` after `yield()`, so `R0` is not the instant its first read finished.

- **location:** `scripts/utilities/metal_abort_session.jl:134-137`
- **kind:** defect
- **found:** #365
- **evidence:**
  The docstring of `reader` (`:117-123`) says that `R0` is the instant the first read finished, and the
  loop assigns it after `sum(v)`, `yield()` and `reads += 1`. When `yield()` hands the thread to another
  task for a long stretch, `R0` lies later than the read, and `T0 < R0 < T1` can fail for a run whose
  first read overlapped the training call, so the run counts as void and not as concurrent. The error
  is on the side of a void run. Read from the code; not measured.

### K39 · The session arm shares `M8_T0`, `M8_T1` and `M8_TRAINER` between two threads with no synchronization.

- **location:** `scripts/utilities/metal_abort_session.jl:102-106`
- **kind:** not verified
- **found:** #365
- **evidence:**
  `training()` assigns the three globals and `reader()` polls and reads them (`:127`, `:133`,
  `:141-142`) on another thread, with no lock and no atomic. Julia gives no guarantee for such reads of
  a global that another thread writes. CodeRabbit marks the finding as one it could not verify, and no
  wrong value has been observed: each global is a single `Int`, assigned around the training
  call.

### K42 · With small weights a `ResNetLayer` integrates only an affine field, and its docstring does not say so.

- **location:** `src/layers/resnet.jl:6`
- **kind:** docs
- **found:** 2026-11-04
- **evidence:**
  This is a property of the form of the layer and not a defect. `x + σ(Wx + b)` has one affine map,
  and it sits inside the activation. For `σ ∈ C²` with `σ(0) = 0` and `σ'(0) = 1`, `W = hA` and
  `b = hc`, the argument of `σ` is small, so `σ` is linear there and each layer is an Euler step of
  `Ax + c`. The iteration over `L = T / h` layers converges with order one to the flow of that
  field, and the limit does not depend on `σ`:

  ```julia
  using GeometricMachineLearning, LinearAlgebra
  using GeometricMachineLearning: ResNetLayer
  import Random
  Random.seed!(1)
  d, T = 4, 1.0
  A, c, x₀ = randn(d, d) / 2, randn(d), randn(d)
  flow = (exp([A c; zeros(1, d + 1)] * T) * vcat(x₀, 1))[1:d]
  function residual_iteration(σ, L)
      l, h, x = ResNetLayer(d, σ), T / L, x₀
      ps = (weight = h * A, bias = h * c)
      for _ in 1:L
          x = l(x, ps)
      end
      x
  end
  [norm(residual_iteration(tanh, L) - flow) for L in (2^10, 2^12, 2^14)]
  # identity [0.00129, 0.000321, 8.03e-5]
  # tanh     [0.00129, 0.000322, 8.04e-5]
  # sin      [0.00129, 0.000322, 8.04e-5]
  ```

  The hypotheses on `σ` carry the statement. `softplus` has `σ(0) ≠ 0` and diverges from the flow
  (`1240`, `4950`, `19800` for the same `L`). With `σ'(0) = 2` the limit is the flow of `2(Ax + c)`.
  `relu` is not differentiable at `0`; it is homogeneous, so the limit is the flow of the
  field `relu.(Ax + c)`, which is piecewise affine (`2.5e-5` at `L = 2^14`).

  `WideResNetLayer` has a second matrix outside the inner activation, and it is consistent with a
  nonlinear field. With `upscale_weight = U`, `upscale_bias = u`, `downscale_weight = hV` and
  `bias = hc`, the layer converges with order one to the flow of `V σ(Ux + u) + c`: for `tanh`, with
  `U, u, V = randn(8, 4), randn(8), randn(4, 8) / 2` drawn after the snippet's `x₀`, the error at
  `T = 3` is `3.9e-3` at `L = 2^8` and `6.2e-5` at `L = 2^14`, and the flow of the field linearised
  at `x₀` misses that flow by `4.5`. `ResNet(dim, n_blocks, activation)` takes the plain
  `ResNetLayer`, because its `width` is `dim`. Closing this entry means a sentence in the docstring
  of `ResNetLayer` that points to `WideResNetLayer`.

### K45 · The schedule of `AdamOptimizerWithDecay` advances per minibatch, and ten call sites in tutorials and scripts and three in the tests give it a horizon in epochs, so it ends before the training does.

- **location:** `docs/src/tutorials/symplectic_autoencoder.md:149`
- **kind:** defect
- **found:** 2026-11-04
- **evidence:**
  The positional argument of `AdamOptimizerWithDecay` is the horizon `n` of `DecayingStatic`, in
  optimization steps, and `train!` takes one step per minibatch. The schedule is
  `α(t) = η₁ (η₂ / η₁)^(t / n)` with no floor, so a horizon in epochs reaches `η₂` early and the step
  goes on falling. The batches per epoch are `number_of_batches` of each site's loader and batch;
  the last step is the formula at the last step of the run, with `η₁ = 1e-2` and `η₂ = 1e-6`.

  | call site | `n` | epochs | batches per epoch | `n` reached at epoch | last step |
  |:--|:--|:--|:--|:--|:--|
  | `docs/src/tutorials/symplectic_autoencoder.md:149` | 65536 | 262144 | 2 | 32768 | `1e-34` |
  | `docs/src/tutorials/symplectic_autoencoder.md:284`, `:480` | 65536 | 65536 | 2 | 32768 | `1e-10` |
  | `docs/src/tutorials/volume_preserving_transformer_rigid_body.md:202` | 500000 | 500000 | 8 | 62500 | `1e-34` |
  | `docs/src/tutorials/symplectic_transformer.md:66` | 2000 | 2000 | 2 | 1000 | `1e-10` |
  | `docs/src/tutorials/linear_symplectic_transformer.md:66` | 2000 | 2000 | 2 | 1000 | `1e-10` |
  | `scripts/reproduction/volume_preserving_transformer/rigid_body.jl:54` | 500000 | 500000 | 8 | 62500 | `1e-34` |
  | `scripts/reproduction/symplectic_autoencoders/online_transformer_for_sae.jl:54` | 65536 | 65536 | 2 | 32768 | `1e-10` |
  | `scripts/reproduction/symplectic_autoencoders/online_sympnet.jl:42`, `:110` | 262144 | 262144 | 2 | 131072 | `1e-10` |
  | `test/optimizers/adam_with_learning_rate_decay.jl:38` | 2048 | 2048 | 13 | 158 | `1e-54` |
  | `test/optimizers/adam_with_learning_rate_decay.jl:66` | 128 | 128 | 13 | 10 | `1e-54` |
  | `test/optimizers/structured_array_parameters.jl:37` | 2 | 2 | 19 for `Batch(5)`, 15 for `Batch(5, 3)` | 1 | `1e-78`, `1e-62` |

  `docs/src/optimizers/optimizer.md:23` shows the same call with no data set, so it has no row.

  The fix is in the schedule. A planned pair of pull requests makes the schedule advance per epoch,
  so that `n_epochs` is the horizon. No call site changes but
  `docs/src/tutorials/symplectic_autoencoder.md:149`, which gives 65536 for a run of 262144 epochs.
  These results depend on the fix:

  - The autoencoder and rigid-body pages show the runs behind the committed weights of
    `docs/src/tutorials/*.h5`, and the scripts are the record of those runs. After the fix a script
    gives other weights at its next run, which takes hours on a GPU. The two transformer pages
    retrain at each build, so their figures change at the next build.
  - The autoencoder page and the scripts do not agree on the runs. For the autoencoder, line 149
    names `integrator_train_epochs`, a constant the page defines at line 271, and
    `online_sympnet.jl:42` passes `n_epochs`. For the integrator, line 284 and
    `online_transformer_for_sae.jl:54` give 65536, and `online_sympnet.jl:110` gives 262144.
    `scripts/utilities/convert_jld2_to_h5.jl:32` converts `sae_parameters.jld2`, which only
    `online_sympnet.jl:67` saves, and both scripts save `integrator_parameters.jld2`. The `.jld2`
    files are not in the repository, and nothing in the tree records the schedule of those runs.
  - `test/optimizers/adam_with_learning_rate_decay.jl:48` orders the final losses of the decayed and
    the plain run, so that test runs again after the fix and its bound can move. The comment at
    lines 57 to 59 says that a 32-epoch budget reaches `η₂` almost at once, which holds per
    minibatch and not per epoch, so it changes with the fix.
    `test/optimizers/structured_array_parameters.jl:41` asserts only finite losses.

  Closing this entry means the per-epoch schedule, the test run again, a correction of that comment,
  and a correction of `docs/src/tutorials/symplectic_autoencoder.md:149`.

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

- **location:** `ReducedComplexityModeling` `src/data_loader/batch.jl:186`
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

### K35 · Neither Metal abort of 2026-09-22 reproduces in a counted run: the `LASympNet(4)` arm with a second evaluation holds one death with the item's signature outside its counted window and stays open, and the `VolumePreservingFeedForward(4)` abort does not reproduce in 59 fresh processes.

- **location:** `scripts/utilities/metal_abort_runs.jl`
- **kind:** upstream
- **found:** 2026-10-27
- **evidence:**
  A `LASympNet(4)` process died once from `MTLReleaseAssertionFailure` in
  `-[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]` while a second session `ex` read a Julia
  vector, and a `VolumePreservingFeedForward(4)` process once from
  `BUG IN CLIENT OF LIBMALLOC: memory corruption of free block` inside the host LLVM JIT with no
  second `ex`. Both ran `Float32` on `MetalBackend()`, Adam, and died in the first training call.
  Neither recurred in the 24 warm runs that followed them, and the arms below ask whether either
  recurs in a fresh process.

  `scripts/utilities/metal_abort_runs.jl`, at the commit `553ba5c` that the runs used, runs one
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
  | `LASympNet(4)` with a concurrent evaluation | two `ex` calls in a private Kaimon session | 16 counted, open | 16 | 0 | 1, the death of run 18 |

  Every counted run of the two arms a shell process runs compiled inside the timer (`compile_time`
  15.6 s to 25.3 s for `LASympNet(4)`, 11.6 s to 29.1 s for `VolumePreservingFeedForward(4)`),
  reported a finite `Float32` loss history, showed `Metal API Validation Enabled`, and had a pid no
  other run used. The crash reporter wrote no new `julia-*.ips` at any point in the 118 runs.

  The stack of the runs is Julia 1.13.3, this package at `1d2faead`, `GeometricOptimizers` 0.9.1,
  Metal.jl 1.11.1, GPUCompiler 2.10.0, KernelAbstractions 0.9.45 and LLVM.jl 9.13.2. The stack of the
  2026-09-22 aborts was Julia 1.13.0, Metal.jl 1.11.1, GPUCompiler 2.8.1, KernelAbstractions 0.9.42
  and LLVM.jl 9.13.1: **the same Metal.jl**, whose `flush_batched_queues!` walks every active
  batched queue and flushes it with no lock, the caller's task's or another task's. Metal.jl 1.12.0
  is not selectable from this environment: `Pkg.status(; outdated = true)` prints
  `Metal v1.11.1 (<v1.12.0)` with no constraint beside it and `KernelAbstractions v0.9.45
  (<v0.10.3)` restricted by this package, `GeometricOptimizers`, `AbstractNeuralNetworks` and
  Metal.jl, and which of the two holds the other down is not established.

  **The third arm is open.** Its concurrency is a second request to one
  session, which no script can make: `scripts/utilities/metal_abort_session.jl` holds its two halves,
  the second being the reading half, which reads a host `Vector{Float32}` of 10⁶ entries in a loop,
  and the script's header gives the procedure. Sixteen of its runs are
  counted, all clean. Run 18, in a private Kaimon session, died with `SIGSEGV` and
  left `julia-2026-10-28-103001.ips` under that run's pid, 49451. It is recorded as one death with
  the item's signature in the load phase, outside the arm's counted window, and not as an abort of
  the arm: the report holds no training frame and no reader frame, one thread is in
  `typeinf_ext_toplevel` under nested top-level evaluations, and the run died before the
  `T0 < R0 < T1` stamps, so nothing shows that a read overlapped the training call. The death is on
  the task that called `device_synchronize` from a top-level evaluation:
  `AGX::ComputeContext…::endComputePass`,
  `-[AGXG16XFamilyBlitContext deferredEndEncoding]`,
  `-[AGXG16XFamilyCommandBuffer commitEncoder]` and
  `-[AGXG16XFamilyCommandBuffer commit]` under Metal.jl's `commit_with_queue_key!`
  (`lib/mtl/command_buf.jl:258`), `flush!` (`src/command_batching.jl:377`) and
  `flush_batched_queues!` (`:392`, `:395`), called from `device_synchronize`
  (`src/synchronization.jl:166`). That is the signature of the `LASympNet(4)` item — a
  command-encoder frame below a Metal.jl frame, as the 2026-09-22 assertion's
  `setCurrentCommandEncoder` was — and the code path of the `julia-2026-10-26-225001.ips` report: a
  task's `device_synchronize` flushes every active batched queue, another task's open command buffer
  among them, and Metal.jl 1.11.1 takes no lock there. That reading is a hypothesis: the stack shows
  where the process died and not which task owned the command buffer, and no evidence of queue
  ownership or of the calls before the death exists. No GeometricMachineLearning frame is in the
  stack, which does not by itself place the cause outside this package.

  The arm stays open, and its record is runs 2 to 18 of the `lasympnet-concurrent` arm: 16 counted,
  16 clean, 0 aborts counted, and 1 void, the recorded death of run 18, which `audit` gives as
  `death with a report of this pid and no record of overlap`; about five minutes a run. Run 18 died
  before it printed a stamp line, so its log records `concurrent=crash_before_stamps` and rebuilds
  its `M8-DEVICE` line from the crash report. The death came while another Julia process compiled
  in this package on the same machine, which the record states because the load can change how the
  two tasks interleave. The session script appends `M8-TRAIN-START` and `M8-READ-START` to the run's
  log file before the timed call can finish, because a session's output is lost when it dies, and
  `audit` counts a death of this arm as its abort only when the two
  records show the overlap.

  An upstream issue for Metal.jl, on the unlocked `flush_batched_queues!`, is drafted and not filed.

  A no-reproduction holds for the two arms a shell process runs, on that stack, and says nothing
  about a true rate below one in twenty, nor about the `VolumePreservingFeedForward(4)` item under a
  second evaluation, which these runs did not test.

### K43 · On a batch of more than one sample, `FeedForwardLoss` through a `SymbolicPullback` and through a `ZygotePullback` descends two objectives.

- **location:** `src/data_loader/train.jl:36`
- **kind:** upstream
- **found:** 2026-11-04
- **evidence:**
  `train!` takes any `AbstractPullback` as `loss`, so a `SymbolicPullback` reaches it, and a
  `SymbolicPullback` of SymbolicNeuralNetworks 0.9.1 trains a network here. This package does not
  define or re-export that type. `FeedForwardLoss` is `_norm(prediction - output) / _norm(output)`
  over the whole batch (AbstractNeuralNetworks 0.9.3, `src/losses.jl:86`), which is not a sum over
  samples. `SymbolicPullback` differentiates the loss of one sample and sums over the batch, and
  the page `limitations.md` of SymbolicNeuralNetworks says so.

  ```julia
  using GeometricMachineLearning, LinearAlgebra, SymbolicNeuralNetworks
  using GeometricMachineLearning: params, ZygotePullback
  using AbstractNeuralNetworks: Dense, FeedForwardLoss
  import Random
  Random.seed!(2)
  nn = NeuralNetwork(Chain(Dense(2, 4, tanh), Dense(4, 2, identity)), CPU(), Float64)
  ps = params(nn)
  symbolic = SymbolicPullback(SymbolicNeuralNetwork(nn), FeedForwardLoss())
  zygote = ZygotePullback(FeedForwardLoss())
  x, y = randn(2, 1, 8), randn(2, 1, 8)
  flat(g) = vcat(vec(g.L1.W), vec(g.L1.b), vec(g.L2.W), vec(g.L2.b))
  gradient(pb, r) = flat(pb(ps, nn.model, (x[:, :, r], y[:, :, r]))[2](1.0))
  per_sample(r) = sum(gradient(zygote, i:i) for i in r)
  r = 1:8
  gs, gz = gradient(symbolic, r), gradient(zygote, r)
  norm(gs) / norm(gz), norm(gs - gz) / norm(gz), norm(gs - per_sample(r)) / norm(gs)
  ```

  | `b` | `‖gs‖ / ‖gz‖` | `‖gs - gz‖ / ‖gz‖` | `‖gs - Σ per sample‖ / ‖gs‖` |
  |:--|:--|:--|:--|
  | 1 | 1.0 | 0.0 | 0.0 |
  | 2 | 3.33 | 2.41 | 5.98e-17 |
  | 8 | 31.9 | 31.0 | 1.49e-16 |

  So the two agree at `Batch(1)`, and above it `gs` is the gradient of `Σᵢ ‖eᵢ‖ / ‖yᵢ‖`, the
  objective with `b = 1`, while `gz` is the gradient of `‖E‖ / ‖Y‖`. Three things follow for a
  caller of `train!`:

  - The value of the pullback stays the loss of the whole batch: `1.3648…` from both in the snippet.
    The history of `train!` with a `SymbolicPullback` therefore reports an objective that the steps
    do not descend.
  - The gradient is a sum and not a mean, so its size grows with `b`, and a step size tuned at one
    batch size does not carry over to another.
  - The minimisers differ. For a constant prediction `c` and the targets `1` and `100`, the whole
    batch has its minimum at `c = 50.5` and the sum over samples at `c = 1`, because the second
    weighs each residual by its own target.

  `train!` with a `NetworkLoss` trains through a `ZygotePullback`, which is the whole-batch
  objective. The `SymbolicPullback` methods of HamiltonianNeuralNetworks are not checked here.

## The test suite

### K3 · `test/quality/jet.jl` does not see an unstable kernel-launch argument

- **location:** `test/quality/jet.jl`
- **kind:** missing test
- **found:** 2026-10-01
- **evidence:**
  Julia 1.13.2, `mutate.jl <repository> <file> <from> <to> quality/jet.jl`, one mutant per run.
  Two mutants that put `Base.inferencebarrier` on an argument of a kernel launch SURVIVED, with no
  assertion failing: `kernel!(c, Base.inferencebarrier(a), b, ndrange = size(c))` in
  `src/kernels/tensor_tensor_mul.jl`, and `tensor_mat_skew_sym_assign_k!(C,
  Base.inferencebarrier(Z), A, ndrange = size(C))` in
  `src/kernels/inverses/tensor_mat_skew_sym_assign.jl`. At
  `target_modules = (GeometricMachineLearning,)` a barrier on a launch argument gives 0 reports, and
  the same barrier in an `@assert` of the same launcher is CAUGHT: the kernel functor takes
  `args...`, so the dispatch sits in KernelAbstractions' frames, which the filter drops. A barrier
  on a value inside the kernel body of `tensor_tensor_mul_kernel!` is CAUGHT.

  The loader's gather launch is outside this file's reach. Its launcher,
  `gather_kernel!(backend)(gathered, data, indices, offset, ndrange = size(gathered))`, is
  `ReducedComplexityModeling`'s `src/data_loader/batch.jl`, which `target_modules =
  (GeometricMachineLearning,)` filters out, and gating it belongs to that package's
  `test/quality/jet.jl`.

### K36 · Three files of `test/data_loader/` test code that is no longer in `src/data_loader/`.

- **location:** `test/data_loader/batch.jl`
- **kind:** defect
- **found:** #365
- **evidence:**
  `src/data_loader/` holds `train.jl` alone, since the loader is `ReducedComplexityModeling`'s. Three
  test files of `test/data_loader/` have no source file of that name:
  `batch.jl` tests the `Batch{:Transformer}` method of `integrate`, which is in
  `src/architectures/transformer_integrator.jl`; `data_loader.jl` tests `solve!` of a `PSDArch`,
  which is in `src/architectures/psd.jl`; and `parametric_data_loader.jl` tests `train!` on a
  `ParametricDataLoader`, which is in `src/data_loader/train.jl`. The layout rule is that
  `test/<path>.jl` tests `src/<path>.jl`; `test-layout.jl --check` prints nothing for the branch, so
  the check does not see the three. Moving them to `test/architectures/transformer_integrator.jl`,
  `test/architectures/psd.jl` and `test/data_loader/train.jl` also changes the lines that list them
  in `test/runtests.jl`.

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

### K44 · No test checks the Jacobian symplecticity of `GSympNet`, `LASympNet`, the linear, activation and bias layers, linear and inverse symplectic attention or the two symplectic transformers.

- **location:** `test/architectures/sympnet_integrator.jl:28`
- **kind:** missing test
- **found:** 2026-11-04
- **evidence:**
  `test/architectures/sympnet_integrator.jl:28` asserts `product_final ≉ 0 ≉ product₀`, with the
  product the bilinear form `𝕁` of two points. A symplectic map preserves that form on tangent
  vectors and, unless it is linear, not on points, and the test only asserts that both values are
  not zero. A map that is not symplectic passes it.

  | what | Jacobian check |
  |:--|:--|
  | `GradientLayerQ`, `GradientLayerP` | a shear pair in `test/layers/sympnet_upscaling.jl:39` (group `slow`), and the decoders of `test/architectures/symplectic_autoencoder_tests.jl` and `test/architectures/weight_tied_symplectic_autoencoder.jl` |
  | `SymplecticAttentionQ`, `SymplecticAttentionP` | `test/layers/symplectic_attention.jl:46` |
  | `GSympNet`, `LASympNet` | none |
  | `LinearLayerQ`, `LinearLayerP`, `ActivationLayerQ`, `ActivationLayerP` | none: `test/layers/sympnet_layers_test.jl` checks tensor slices |
  | `LinearSymplecticAttentionQ`, `LinearSymplecticAttentionP` | none: `test/layers/linear_symplectic_attention.jl` compares the output with its formula |
  | `BiasLayer`, `InverseSymplecticAttentionQ`, `InverseSymplecticAttentionP`, `LinearSymplecticTransformer`, `SymplecticTransformer` | none |

  The pinned outputs of `test/layers/layer_pins.jl` and `test/architectures/architecture_pins.jl`
  see a change of a layer and not the property. The layers of the snippet below are symplectic, so
  for them this entry is a gap in the suite and not a defect of the code:

  ```julia
  using GeometricMachineLearning, ForwardDiff, LinearAlgebra
  using GeometricMachineLearning: params, LinearLayerQ, LinearLayerP, ActivationLayerQ,
                                  ActivationLayerP
  import Random
  Random.seed!(4)
  Ω = [zeros(2, 2) I; -I zeros(2, 2)]
  for c in (GSympNet(4), LASympNet(4), Chain(LinearLayerQ(4)), Chain(LinearLayerP(4)),
      Chain(ActivationLayerQ(4, tanh)), Chain(ActivationLayerP(4, tanh)))
      nn = NeuralNetwork(c, CPU(), Float64)
      J = ForwardDiff.jacobian(x -> nn.model(x, params(nn)), randn(4))
      println(nameof(typeof(c)), " ", maximum(abs, J' * Ω * J - Ω))
  end
  ```

  ```
  GSympNet 5.847383476376134e-17
  LASympNet 4.440892098500626e-16
  Chain 0.0
  Chain 0.0
  Chain 0.0
  Chain 0.0
  ```

  The snippet does not reach the two linear attention layers, which take a sequence. With the
  sequence form of `test/layers/symplectic_attention.jl:37` the check gives `0.0` for both.
  `LASympNet` holds a `BiasLayer`. The inverse attention layers and the two transformers are not
  measured.

  Closing this entry means replacing the assertion of `test/architectures/sympnet_integrator.jl`
  with a check of this form for each row marked none.

### K47 · A single `@allocated` reading in the batch-scaling checks can carry a stray allocation, so the checks fail at random.

- **location:** `test/layers/bilinear_and_low_rank_stiefel.jl:848`
- **kind:** defect
- **found:** #393
- **evidence:**
  The 13 `@allocated` readings of lines 810–855 are one reading per batch size, and
  `n_independent` compares them with a tolerance of 4096 B. Now and then one reading carries an
  extra allocation of fixed size: 124640 B on Julia 1.12.7, 938976 B on Julia 1.13.4. That is
  enough to fail a check wherever it lands. In 12 fresh runs of the file on 1.12.7 with
  `--check-bounds=auto` on macOS, 3 failed (lines 848, and 859 with 861); in 3 runs on 1.13.4, 1
  failed (line 834). CI on #393 failed line 848 on Julia min, ubuntu and windows:

  ```
  Batched Zygote differentiation and allocation scaling: Test Failed at test/layers/bilinear_and_low_rank_stiefel.jl:848
    Expression: n_independent(alloc_loss_bwd2 / B2, alloc_loss_bwd1 / B1)
  ```

  In a steady state every check passes with room, the same on 1.12.7 with `yes` and `auto` and on
  1.13.4: line 848 gives 563 B (Float32) and 612 B (Float64), and the largest value is 1186 B
  (line 859, Float64). So neither the bounds-check setting nor the code under test causes the
  failure; 80000 readings of the `loss` gradient alone show no spike. A comparison of slopes over
  three sizes fails on the spike too. The minimum of a few readings per size passes in every run
  measured. Allocation checks of the same form in other packages of this tree can fail the same
  way, and their fix is to be decided together.
