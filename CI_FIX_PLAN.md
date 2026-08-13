# Workplan: fixing CI on PR #230 and closing issue #234

Status updated on 2026-08-13 on branch `use-geometric-optimizers`. CI run
`31570166729` had completed its Julia 1.12 builds but remained in
`julia-runtest`; R1, R3, and R8 therefore remain unresolved CI blockers.

Applied fixes cover the test policy, documentation cleanup, CI progress
visibility, generated-artifact hygiene, and the local R8 dispatch mitigation.
The remaining work is either upstream release/registration, CI confirmation,
or the separate issue #234 design change; the canonical actions are listed in
§3 and the verification state is tracked in §7.

The R8 mitigation changes the GO-backed leaf update in `src/utils.jl` from an
`isa` branch to concrete optimizer-method dispatch. Offline checks pass for
bibliography uniqueness, Julia parsing, stale-reference scans, and
`git diff --check`; runtime validation remains blocked by the local read-only
Julia depot and unavailable network access. Phase 0 is likewise blocked until
the local `GeometricOptimizers` `0.2.0` checkout is tagged and registered.

---

## 1. Where we stand

| Job | Result | Root cause |
| --- | --- | --- |
| Julia 1.10 × {ubuntu, macOS, windows} | ❌ | **R1** unresolvable `GeometricOptimizers` |
| Julia 1.12 × {ubuntu, macOS, windows} | ⏳ still running after > 1 h 15 min (vs 29 min for the whole job on `main`) | **R8** pathological compile time in the GO-backed optimizer path (all three builds passed; `julia-runtest` does not finish) |
| Julia ^1.13.0-0 × {ubuntu, macOS, windows} | ⚠️ experimental | **R3** precompile |
| Julia nightly × {ubuntu, macOS, windows} | ❌ (`experimental: true`) | **R3** precompile |
| Documentation | ⏳ | R4 fixed; R5/R6 docs updates applied, full build pending |
| PDF | ⏳ | R4 fixed; full build pending |

The documentation issues that were latent behind R4 are now addressed:

* **R4** duplicate bibliography keys — removed.
* **R5** stale `@docs` blocks — removed or limited to GML-owned symbols.
* **R6** stale optimizer constructors — tutorial call sites updated to v0.2.

And one design defect, tracked separately:

* **R7** = issue #234: GML's manifolds do not subtype `GeometricOptimizers.Manifold`,
  so GO's generic retractions never dispatch.

### Julia 1.12 local investigation (2026-08-12) (analyzed by codex)

The requested Julia 1.12.6 run was repeated with a writable temporary depot.
Dependency setup completed, including `GeometricOptimizers`; doctests and PSD
tests passed. The apparent stall begins in the next `symplectic_autoencoder`
testset, but it is compilation rather than a training loop deadlock:

* The first optimizer epoch takes about 13 seconds for compilation, then each
  later epoch takes about 7 ms; 100 epochs complete and reach the expected
  accuracy threshold.
* An instrumented `test_symplecticity(10, 6)` completes in about 18 seconds:
  the first Jacobian takes about 7.6 seconds, training for 10 epochs about
  8.7 seconds, and the second Jacobian about 0.3 seconds.
* Calling the same optimizer code from the test helper function can spend
  several minutes in Julia 1.12's compiler before producing any output. An
  interrupt shows billions of allocations in `Compiler` inference and
  `subtype.c`; changing `N::Integer`/`n::Integer` to `Int` does not resolve it.

Thus the 1.12 latency is a Julia 1.12 inference/compiler pathological case
triggered by the optimizer call nested in the test helper, amplified by the
silent `SafeTestsets` output. No production-code defect or test assertion
failure has been reproduced yet.

### Julia 1.12 local investigation II (2026-08-12) (analyzed by claude)

The second investigation added evidence that is useful for prioritising R8,
but does not change the root cause or action list:

* This is a regression relative to `main`, not merely a slow Julia 1.12 job:
  the 1.12 jobs on `main` completed in roughly 25–48 minutes, while PR run
  `31570166729` was still in `julia-runtest` after 1 hour 15 minutes on all
  three operating systems.
* Two `SIGINFO` samples showed the compiler spending its time in
  `subtype_unionall`, `subtype_tuple_varargs`, `jl_type_union`, and method
  intersection. That points to signature shape and method-table enumeration,
  rather than inference of a large function body.
* The leading upstream hypothesis is GO's union aliases used as type-parameter
  bounds on optimizer cache/state structs, especially aliases containing
  `NamedTuple` and `Vararg` members. This remains a hypothesis until profiling
  or an upstream reproduction confirms it.
* The same SAE workload is fast when invoked at top level (about 13–18 seconds
  for a cold first epoch and milliseconds when warm, including the
  `--check-bounds=yes -g1` case). The pathological latency appears when the
  optimizer is compiled through the test helper, so changing test structure
  would only hide a user-facing issue.
* The independent run did not complete the full 1.12 suite, so the checklist
  still correctly requires one complete end-to-end run before declaring R8
  resolved.

## 2. Root causes in detail

### R1 — `GeometricOptimizers` cannot be resolved on Julia 1.10

```
ERROR: Unsatisfiable requirements detected for package GeometricOptimizers [fc236c15]:
 possible versions are: 0.1.0 or uninstalled; restricted to versions 0.2
```

`Project.toml` pins `GeometricOptimizers = "0.2"`, but General only carries
0.1.0. The PR works around this with

```toml
[sources]
GeometricOptimizers = {url = "https://github.com/JuliaGNI/GeometricOptimizers.jl.git", rev = "main"}
```

`[sources]` is honoured only by Pkg ≥ 1.11, so all three 1.10 jobs die at
`Pkg.instantiate`. Secondary problem: **General rejects any package with a
`[sources]` block**, so as long as this stays in, GML itself cannot be
registered either. Third: GML declares `julia = "1.9"` while GO declares
`julia = "1.10"` — inconsistent even once resolution works.

### R2 — Doctests are sensitive to the RNG stream (Julia 1.13)

`src/data_loader/batch.jl` has two jldoctests that hardcode the output of
`shuffle` (lines 144–145: `shuffle(1:number_columns)` /
`shuffle(1:third_dim)` on the global `TaskLocalRNG`). The stream changed
between 1.12 and 1.13:

| Doctest | Expected | 1.13 produced |
| --- | --- | --- |
| `batch.jl:18-34` | `([(1, 5), (1, 3)], [(1, 4), (1, 1)], [(1, 2)])` | `([(1, 3), (1, 1)], [(1, 5), (1, 4)], [(1, 2)])` |
| `batch.jl:106-130` | `([(1, 1), (4, 1), (2, 1)], [(3, 1)])` | `([(1, 1), (3, 1), (2, 1)], [(4, 1)])` |
| `batch.jl:106-130` | `([(1, 3), (1, 2), (1, 4)], [(1, 1), (1, 5)])` | `([(1, 5), (1, 4), (1, 1)], [(1, 3), (1, 2)])` |

This became a *test-suite* failure (not just a docs failure) because the PR
re-enabled the doctest testset in `test/runtests.jl` — on `main` that line is
commented out. It ran first, so it aborted the whole suite before any real
test executed.

Appears to be 1.13-only: codex observed the doctest testset **passing** on 1.12
before the suite stalled in R8, and my stack samples show a `@safetestset`
module already under evaluation, which can only happen after the doctest
testset has returned. So R2 and R8 look independent, and fixing R2 will not
move the 1.12 jobs.

**Resolution on August 12, 2026:** Documenter doctests are no longer part of
`test/runtests.jl`; they remain owned by the documentation workflow. The
examples are covered by ordinary unit tests under `test/`, with duplicate
coverage integrated into existing array, kernel, loader, layer, and attention
test files. Standalone example tests remain only for APIs without a natural
existing test home. This removes the RNG-sensitive doctest failure from the
Julia 1.13 unit-test matrix without sacrificing behavioral coverage.

### R3 — Method overwriting during precompilation (Julia ≥ 1.13)

`GenericLinearAlgebra v0.4.0` redefines
`LinearAlgebra.eigencopy_oftype(::UpperHessenberg, S)`, which the stdlib
already defines (`hessenberg.jl:483`). Julia ≥ 1.13 makes this a hard error
during precompilation. Dependency chain:

```
GeometricIntegrators v0.17.0 → RungeKutta v0.5.23 → GenericLinearAlgebra v0.4.0
```

`GeometricIntegrators` is a *test-only* dependency (`[targets] test`).
GenericLinearAlgebra's master branch guards the definition with
`VERSION < v"1.14.0-DEV.2266"`, i.e. it is still broken on 1.13. **This cannot
be fixed inside GML** other than by constraining/removing the test dependency.

### R4 — Duplicate BibTeX keys (resolved)

The duplicate `Kraus:2020:GeometricIntegrators` and
`greydanus2019hamiltonian` entries were removed from
`docs/src/GeometricMachineLearning.bib`. Keep the bibliography uniqueness
check in the verification checklist until both documentation workflows are
observed green.

### R5 — `@docs` blocks pointing at symbols that no longer exist

| File | Entries that will fail |
| --- | --- |
| `optimizers/optimizer_framework.md` | `Optimizer` (new struct in `src/utils.jl` has **no docstring**), `optimization_step!`, `optimize_for_one_epoch!` |
| `optimizers/optimizer_methods.md` | `OptimizerMethod`, `GradientOptimizer`, `MomentumOptimizer`, `AdamOptimizer`, `AbstractCache`, `GradientCache`, `MomentumCache`, `AdamCache` (all GO symbols / aliases → docstrings live in GO, which is not in `modules=[...]`), `update!(::Optimizer, ::AbstractCache, ::AbstractArray)` (method gone), `AdamOptimizerWithDecay` (exists, no docstring) |
| `optimizers/bfgs_optimizer.md` | `BFGSOptimizer`, `BFGSCache`, `update!(::Optimizer{<:BFGSOptimizer}, ::BFGSCache, ::AbstractArray)` — **BFGS no longer exists in GML at all**; GO exposes it as `_BFGS()` via a completely different entry point (`Optimizer(x, problem; algorithm=_BFGS())`). The page also has live `@example` blocks calling `BFGSCache(B̄)`. |
| `optimizers/manifold_related/retractions.md` | `GeometricMachineLearning.geodesic(::StiefelLieAlgHorMatrix)`, `…geodesic(::GrassmannLieAlgHorMatrix)`, `…cayley(::StiefelLieAlgHorMatrix)`, `…cayley(::GrassmannLieAlgHorMatrix)`, `…cayley(::Manifold{T}, ::AbstractMatrix{T})`, `GeometricMachineLearning.𝔄(::AbstractMatrix)`, `…𝔄(::AbstractMatrix, ::AbstractMatrix)` — `𝔄` exists only as `GeometricOptimizers.𝔄` |
| `introduction.md` | `[`GradientOptimizer`](@ref)` cross-references (×2) |

### R6 — Tutorials call constructor signatures that GO v0.2 removed

GO v0.2 moved the learning rate out of the method and into
`Optimizer(...; step_size=…)`, and deliberately made the old calls fail loudly:

| GO v0.2 reality | Docs still call |
| --- | --- |
| `GradientMethod()` — **no arguments** | `GradientOptimizer(η)` (`optimizer_methods.md:30`, `optimizer_comparison.md:32`), `GradientOptimizer(T(0.001))` (`mnist_tutorial.md:124`) |
| `MomentumMethod(α)` — **one argument** | `MomentumOptimizer(η, α)` (`optimizer_methods.md:66`), `MomentumOptimizer(T(0.001), T(0.5))` (`mnist_tutorial.md:125`) |
| `Adam(::Type{T}=Float64; β₁, β₂, δ)` | `AdamOptimizer(η)` (`optimizer_comparison.md:33`, `grassmann_layer.md:268`), `AdamOptimizer(η, ρ₁, ρ₂, δ)` (`optimizer_methods.md:123`) |
| `AdamOptimizerWithDecay(n_epochs, η₁=1f-2, η₂=1f-6, ρ₁, ρ₂, δ; T=typeof(η₁))` | `AdamOptimizerWithDecay(n_epochs, Float64)` (`linear_symplectic_transformer.md:66`, `symplectic_transformer.md:66`) → binds `η₁ = Float64` (a `DataType`); `AdamOptimizerWithDecay(n_epochs, T; η₁=1e-2, η₂=1e-6)` (`volume_preserving_transformer_rigid_body.md:191`) → no such kwargs |
| `Optimizer(method, nn)` only (`src/utils.jl:268,276`) | `Optimizer(nn, AdamOptimizer(1e-1))` (`grassmann_layer.md:268`), `Optimizer(sae_nn_gpu, AdamOptimizerWithDecay(…))` (`symplectic_autoencoder.md:137`) — **network-first order is gone** |
| BFGS only via `algorithm=_BFGS()` | `BFGSOptimizer(η)` (`optimizer_comparison.md:34`) |

These are `@example` blocks, i.e. they execute during the docs build.

### R7 — Issue #234: duplicated retraction pipeline

GO's generic retraction is

```julia
function geodesic(Y::Manifold{T}, Δ::AbstractMatrix{T}) where {T}
    λY = GlobalSection(Y); B = global_rep(λY, Δ); E = StiefelProjection(B)
    expB = geodesic(B); λY * typeof(Y)(expB * E)
end
```

GML's `StiefelManifold`/`GrassmannManifold` subtype **`GeometricMachineLearning.Manifold`**,
a distinct abstract type, so this method never applies. Commit `b16267ea`
worked around it by re-implementing the exact same pipeline four times
(`geodesic`/`cayley` × `Stiefel`/`Grassmann`) as
`GeometricOptimizers.geodesic(Y::StiefelManifold{T}, …)` etc. That is the
symptom issue #234 reports. The same duplication exists for
`geodesic`/`cayley` on the Lie-algebra-horizontal types and for the
`_copyto!`/`_add!`/`_rac!`/`_square!`/`_div!` family — roughly 30 bridge
methods in total, plus near-verbatim copies of GO's `src/manifolds/*` and
`src/arrays/*`.

### R8 — Pathological compile time in the GO-backed optimizer path

Evidence and analysis are in "Claude's findings" above. In short: inference of
a call that reaches `GeometricOptimizers.update!` through GML's
`_tree_optim_step!`/`_leaf_optim_step!` spins in method-table intersection
(`ml_matches` → `jl_type_intersection_env_s` → `ijl_type_union` →
`subtype_unionall`). The suspected cause is GO's `Union` aliases
(`OptimizerSolution`, `GradientArrayOrNamedTuple`,
`GlobalSectionSingleOrNamedTuple`) being used as *type-parameter bounds* on
every optimizer cache and state struct; this is a hypothesis from the
backtrace, not yet confirmed by profiling.

Two properties make R8 different from R1–R6 and easy to underestimate:

* **It produces no error.** The suite does not fail, it just does not finish.
  In a CI log that is indistinguishable from a hung runner, so it is a merge
  blocker for #230 even though nothing is red yet.
* **It is a user-facing latency bug, not only a test problem.** Every GML user
  who calls `Optimizer`/`optimization_step!` from inside a function — i.e.
  everyone who writes a training script as a function — pays this compile cost.
  Fixing it only in the tests would hide it, not solve it.

It is almost certainly not version-specific: 1.12 is simply the only matrix
entry that currently gets far enough to reach it (1.10 dies at R1, 1.13 at R2).

---

## 3. The plan

Phases 0–1 are the critical path (nothing else can be verified until CI can
resolve dependencies). Phases 2–4 are independent of each other and can be done
in parallel. Phase 5 is a design change that should **not** block merging #230.
Phase 7 (R8) was added after the 1.12 investigation; it **is** a merge blocker
and can be worked on in parallel with everything else, since it needs no
dependency resolution to reproduce.

### Phase 0 — Release GeometricOptimizers v0.2.0 (upstream, blocking)

Owner: whoever holds JuliaGNI release rights. Everything else waits on this.

1. In `../GeometricOptimizers` on `main`: confirm `version = "0.2.0"` in
   `Project.toml`, confirm CI is green, tag and push.
2. Register in General (`@JuliaRegistrator register` on the release commit, or
   via the registration workflow). Verify the version appears in
   `General/G/GeometricOptimizers/Versions.toml`.
3. Sanity-check GO's own `[compat]` — in particular `SimpleSolvers = "0.10"`
   and `julia = "1.10"` — since GML will inherit those constraints.

*If registration cannot happen before #230 needs to merge*, the only honest
alternative is to drop Julia 1.10 from the CI matrix and set
`julia = "1.11"` in `[compat]`, keeping `[sources]`. That trades a CI failure
for a support regression and still blocks GML's own registration, so it should
be a conscious decision, not a default.

**Decided (§6, answer 1): this interim is acceptable**, provided an issue is
opened in GML recording that 1.10 support must be restored once GO v0.2.0 is
registered. In that case Phase 1's `julia = "1.10"` becomes `julia = "1.11"`
and the `[sources]` block stays for now — see the note in Phase 1.

### Phase 1 — Make GML resolvable

`Project.toml`:

* Delete the entire `[sources]` block.
* Keep `GeometricOptimizers = "0.2"` in `[compat]`.
* Bump `julia = "1.9"` → `"1.10"` (GO's floor; 1.9 was never actually
  satisfiable with this dependency set).
* Add compat bounds for any new direct dependency the PR introduced that lacks
  one (check `ParameterHandling`, `SimpleSolvers` if they are now direct).

Expected effect: 1.10 × 3 OS turn green through `instantiate`, and the
`General` registration path for GML is unblocked.

⚠️ This is the *post-registration* form. If Phase 0 has not happened yet and the
interim from §6 answer 1 is taken instead, then the opposite applies: keep
`[sources]`, set `julia = "1.11"`, drop 1.10 from the matrix in
`.github/workflows/CI.yml`, and open the tracking issue. The two variants are
mutually exclusive — pick one deliberately and say which in the PR description.

### Phase 2 — Separate documentation checks from unit tests (completed)

The two `src/data_loader/batch.jl` doctests assert a specific shuffled
permutation. The package test runner now excludes Documenter doctests:

* `test/runtests.jl` contains no `Documenter.doctest` invocation;
* the documentation workflow remains responsible for rendering and doctest
  validation; and
* ordinary tests cover the examples' behavior using structural assertions.

This is strictly better than the alternatives (seeding the RNG inside the
doctest still breaks whenever the RNG stream changes; `doctestfilters` would
hide genuine regressions), while keeping documentation validation in the
documentation workflow.

### Phase 3 — Fix the documentation and PDF builds

**3a. Bibliography (completed).**
The duplicate bibliography keys have been removed. Verify uniqueness as part
of the documentation checks rather than repeating the deletion step.

**3b. `@docs` blocks (R5).**
For symbols that now live in GO, do **not** simply add `GeometricOptimizers`
to `modules=[...]` in `docs/make.jl`: Documenter's default `checkdocs = :all`
would then demand a page for every GO docstring and produce hundreds of new
errors. Instead:

* Rewrite `optimizers/optimizer_methods.md` as prose that describes the
  methods and links out to GO's documentation, keeping `@docs` only for
  GML-owned symbols.
* Add docstrings to the GML-owned symbols that the docs reference:
  `Optimizer` (`src/utils.jl:251`), `AdamOptimizerWithDecay`
  (`src/utils.jl:208`), `optimization_step!`.
* `optimizers/bfgs_optimizer.md`: BFGS is gone from GML. Either delete the page
  (remove it from the `html` nav **and** from the LaTeX page list plus the
  `value_for_key(_optimizers, "Optimizer Methods", "BFGS Optimizer")` entry in
  `docs/make.jl`), or rewrite it around GO's `_BFGS()` entry point. Deleting is
  the lower-risk choice for this PR; note it in the changelog.
* `manifold_related/retractions.md`: drop the `@docs` entries for the four
  Lie-algebra `geodesic`/`cayley` methods, `cayley(::Manifold, ::AbstractMatrix)`
  and the two `𝔄` methods; replace with prose + links. (If Phase 5 lands
  first, several of these can instead point at GO directly.)
* `introduction.md`: replace the two `[`GradientOptimizer`](@ref)` links with a
  symbol that is still documented in GML.

Consider `DocumenterInterLinks` for clean cross-references into GO's docs
rather than plain-text mentions.

**3c. Tutorials (R6).** This is the largest single chunk of work and the one
most likely to be underestimated. Every call site in the table under R6 must be
rewritten to the v0.2 API — learning rates move from the method constructor to
`Optimizer(...; step_size = …)`:

```julia
# before
o = Optimizer(nn, AdamOptimizer(1e-1))
# after
o = Optimizer(Adam(Float64), nn; step_size = 1e-1)
```

**Decided (see §6, answer 2): do not restore the old signatures.** Neither the
network-first `Optimizer(nn, method)` nor the type-first
`AdamOptimizerWithDecay(n_epochs, T::Type; η₁, η₂)` is to be added back;
optimizer functionality belongs in GO, not GML. So *every* call site in the R6
table must be updated in `docs/src/tutorials/` and in the user-facing docs, and
the break must be called out in the release notes / changelog.

Corollary that follows from the same answer and is **not** yet reflected
elsewhere in this plan: `AdamOptimizerWithDecay` itself is currently defined in
GML (`src/utils.jl:208`), not in GO. By answer 3 that means it should either
move to GO or get its own workplan file rather than being quietly kept — decide
this before rewriting the four tutorials that call it, because the target API
depends on the outcome.

**3d.** Re-check `docs/Project.toml` — it must also drop any `[sources]`
workaround once GO v0.2.0 is registered.

### Phase 4 — Julia 1.13 and nightly (R3)

Not fixable inside GML. Options, in order of preference:

1. **Push upstream.** Open an issue/PR on GenericLinearAlgebra to widen the
   version guard so it also covers 1.13 (currently only
   `VERSION < v"1.14.0-DEV.2266"`), and/or on RungeKutta to relax its
   GenericLinearAlgebra bound.
2. **Constrain the test dependency.** Add a `[compat]` entry for
   `GenericLinearAlgebra` (as an `[extras]` dep) pinning a version that does not
   trigger the overwrite, if one exists. Needs checking against RungeKutta's
   own bounds.
3. **Isolate `GeometricIntegrators`.** It is used by only a handful of test
   files. Moving those into a separate test target/workflow keeps the main
   suite runnable on 1.13.
4. **Accept it for now.** Mark `^1.13.0-0` as `experimental: true` in
   `.github/workflows/CI.yml` alongside nightly until upstream is fixed. This
   is the pragmatic choice for merging #230, but it must be paired with option 1
   so it does not become permanent.

Note that 1.13 also fails on R2, so Phase 2 must land regardless — otherwise
option 4 hides two problems instead of one.

### Phase 5 — Issue #234 (do not block #230 on this)

Three ways to make GO's generic retractions apply to GML's manifolds:

**Option A — GML's abstract types alias GO's (recommended near-term).**

```julia
const Manifold = GeometricOptimizers.Manifold
```

GML keeps its own concrete `StiefelManifold`/`GrassmannManifold` types and all
their GML-specific methods, but they now subtype GO's abstract `Manifold`, so
GO's generic `geodesic`/`cayley` dispatch directly. All four bridge methods
from `b16267ea` can be deleted with **no change to GO**.

⚠️ Hazard that must be handled as part of this change: `src/manifolds/abstract_manifold.jl`
is a near-verbatim copy of GO's, so after aliasing, GML's generic methods
(`Base.rand`, `size`, `getindex`, `copy`, the `similar`/`fill!` error methods,
broadcasting, `_round`) would have *identical signatures* to GO's and silently
overwrite them — which on Julia ≥ 1.13 is a hard precompilation error, i.e. it
would re-create R3 from inside GML. So Option A requires deleting GML's copies
of those generic methods and keeping only the ones that genuinely differ (e.g.
`rand` dispatching on GML's `networkbackend`/device types, which needs a
distinct signature). Do the same for `AbstractLieAlgHorMatrix`.

**Option B — a trait in GO.** Give GO a Holy trait so *any* external hierarchy
can opt in:

```julia
abstract type ManifoldTrait end
struct IsHomogeneousSpace <: ManifoldTrait end
struct NotAManifold      <: ManifoldTrait end
ManifoldTrait(::Type)                = NotAManifold()
ManifoldTrait(::Type{<:Manifold})    = IsHomogeneousSpace()

geodesic(Y::AbstractMatrix{T}, Δ::AbstractMatrix{T}) where {T} =
    geodesic(ManifoldTrait(typeof(Y)), Y, Δ)
```

with the generic body factored into a shared `_retract(retraction, Y, Δ)` that
calls `apply_section(λY, …)` instead of `λY * …` (so it no longer depends on
`Base.:*(::GlobalSection, ::GO.Manifold)` either). Choose this only if GML must
keep an independent type hierarchy — it is more machinery than Option A for the
same result, and it changes `geodesic(::AbstractMatrix, ::AbstractMatrix)` from
a `MethodError` into a custom error.

**Option C — full type unification (long-term).** GML stops defining manifold,
Lie-algebra-horizontal and structured-matrix types altogether and re-exports
GO's. This deletes `src/manifolds/*` and much of `src/arrays/*` outright. It is
the right end state, but it is a large refactor touching layers, AD rules,
`networkbackend`, `Ω`, GPU kernels operating on `.A`, and `rand` on GML device
types — and some `Base` methods on GO types would become type piracy. Track it
as its own issue.

**Recommendation:** Option A now (it closes #234 and deletes the `b16267ea`
duplication), Option C as a tracked follow-up.

### Phase 6 — Repo hygiene before merge (completed)

The PR-only workplan and inspection script are gone, the former BFGS test has
been renamed to `test/optimizers/gradient_optimizer.jl`, and the generated
artifact patterns are covered by `.gitignore`. This pass also removes the
remaining tracked MNIST PDFs and the stray `.jld2` file; generated outputs
should not be carried in the source tree.

### Phase 7 — R8: make the optimizer path compile in reasonable time (blocking)

Added after the 1.12 investigation. This is the canonical action list for R8:

1. **Localise it.** `@snoopi_deep` (SnoopCompileCore) on
   `test_accuracy(10, 6; n_epochs = 1)`, or `--trace-compile=stderr` to see the
   last signature compiled before the spin. This decides whether the GO type
   aliases are really the cause or only a plausible one.
2. **Fix it upstream in GO** if confirmed: stop using `OptimizerSolution` /
   `GradientArrayOrNamedTuple` / `GlobalSectionSingleOrNamedTuple` as
   type-parameter *bounds* on the cache and state structs
   (`optimizer_solution.jl:4-26` and the six struct definitions listed in the
   findings). Enforce the invariant in inner constructors instead. Behaviour is
   unchanged, so this can ride along with the Phase 0 release.
3. **Reduce the dispatch fan-out in GML**: `_leaf_optim_step!`
   (`src/utils.jl:316-357`) branches on `adapted isa …` at one call site, so a
   single inference target sees the whole `update!` method table. Ordinary
   dispatch, or a `@nospecialize`d barrier, confines that. **Implemented
   locally** with `_go_update_leaf!` overloads for `Adam`, `MomentumMethod`,
   and the fallback `OptimizerMethod` path.
4. **Do not "fix" it by restructuring the tests.** Moving the optimizer call out
   of the test helper makes CI green while leaving the latency in place for
   every user who wraps training in a function. Acceptable only as a temporary
   unblock, and only with a linked issue.

Success criterion: the 1.12 × ubuntu job returns to the ~30 min range it has on
`main`, and `test_accuracy(10, 6; n_epochs = 1)` compiles in seconds rather
than minutes.

Interaction with §4: if the traversal in `src/utils.jl` is replaced by GO's
native NamedTuple handling (see below), Phase 7 step 3 disappears with it. That
makes the §4 follow-up more attractive than "not now" suggests — it may be the
real fix rather than a cleanup.

---

## 4. Follow-up worth doing, but not now

`src/utils.jl` re-implements optimizer-tree traversal (`_make_optimizer_cache`,
`_make_optimizer_state`, `_tree_optim_step!`) and a bespoke `GMLEuclideanState`
holding `m₁`/`m₂` by hand. GO v0.2 already supports NamedTuple-valued solutions
natively (`GradientArrayOrNamedTuple`, `OptimizerSolution`,
`GlobalSection(::NamedTuple)`, `ParameterHandling.flatten` for `Manifold`), and
already has `AdamCache`/`AdamState`/`MomentumCache`/`MomentumState`. Most of
these 262 new lines are probably replaceable by GO's own machinery. Keep this
as a separate issue unless Phase 7 profiling confirms that the traversal is
the actual R8 trigger; in that case, replacing it with GO's native support is
the smallest correct fix rather than a cleanup-only refactor.

---

## 5. Suggested order of execution

```
Phase 0 (upstream release)  ──┬─→ Phase 1 (Project.toml)  ──→ 1.10 jobs green
                              │
Phase 3a (bib)  ──────────────┼─→ Phase 3b/3c (docs + tutorials) ──→ Documentation + PDF green
                              │
Phase 2 (test-policy) ────────┼─→ 1.13 doctest failures removed from unit CI
                              │
Phase 4 (upstream/CI policy) ─┘   → 1.13/nightly precompile addressed or accepted

Phase 7 (R8 compile time) — independent, BLOCKING; start now, it needs no
                            dependency resolution to reproduce. Step 2 should
                            land in GO before Phase 0 tags v0.2.0.
Phase 5 (#234)  — separate PR
Phase 6 (hygiene) — fold into #230 before merge
```

Note that Phases 0–4 can all land and *every* job will still be red or hanging
until Phase 7 is done, because R8 blocks the only jobs that reach the test
suite. Phase 7 is therefore the one to start first in practice, even though
Phase 0 is the formal critical path.

## 6. Questions raised by claude with answers provided by benedict-96

1. Can GO v0.2.0 be tagged and registered now? If not, is dropping Julia 1.10
   support acceptable as an interim?
2. Restore the backwards-compatible `Optimizer(nn, method)` and
   `AdamOptimizerWithDecay(n_epochs, T::Type)` signatures (smaller docs diff,
   no user-facing break) — or update every call site and document the break?
3. Delete `docs/src/optimizers/bfgs_optimizer.md`, or rewrite it around GO's
   `_BFGS()`?
4. Must `^1.13.0-0` stay a required job while GenericLinearAlgebra is broken on it?
5. For #234: Option A (alias GML's abstract types to GO's) now, with Option C
   (full unification) as a follow-up issue — agreed?

### Answers provided by benedict-96

1. Dropping Julia 1.10 support is an acceptable interim solution. In that case we should open an issue to indicate that 1.10 support has to be restored. We can fix the other CIs first though.
2. We should update every call site to match the current interface. Optimizer functionality should be in `GeometricOptimizers` and not in `GeometricMachineLearning`.
3. See answer above. If we replicate code from `GeometricOptimizers`in `GeometricMachineLearning`, then this should be removed. If optimizer-related functionality is defined in `GeometricMachineLearning`, but not in `GeometricOptimizers`, then pause and add a seprate workplan file for such a task.
4. If `GenericLinearAlgebra` breaks the `^1.13.0-0` support, than this support should not be required for now. But we should open an issue in `GeometricMachineLearning` to indicate that this requirement will have to be restored. Also look into the open issues of `RungeKutta`/`GeometricIntegrators` to check if this problem has already been raised and if it has, please reference this `GenericLinearAlgebra` issue in the `GeometricMachineLearning` issue.
5. Yes, we do A now but need to open an issue for C!

### Further questions raised by claude after the 1.12 investigation (unanswered)

6. R8 step 2 changes GO's cache/state struct signatures. Should that land in GO
   **before** v0.2.0 is tagged (Phase 0), or as a v0.2.1 afterwards? Landing it
   first avoids registering a version with a known compile-time pathology, but
   delays Phase 0, which everything else waits on.
7. Answer 2 says optimizer functionality belongs in GO. Does that verdict extend
   to `AdamOptimizerWithDecay` (currently GML-only, `src/utils.jl:208`) and to
   the `_make_optimizer_cache`/`_tree_optim_step!` traversal in §4 — i.e. should
   they move to GO as part of this PR, or is the separate-workplan route from
   answer 3 the intended path?
8. If Phase 7 turns out to need an upstream Julia fix rather than a GO/GML one,
   is temporarily restructuring the test helpers (step 4) acceptable to get #230
   merged, with a tracking issue — or should #230 wait?
9. Should `runtests.jl` get per-testset progress markers (`@info` before each
   `@safetestset`)? It is unrelated to the fix, but without it the next
   long-running job is again indistinguishable from a hung one.

### Further answers provided by benedict-96

6. If this doesn't change the API we can wait for v0.2.1. If it does please open an issue at the appropriate spot and tag michakraus.
7. All of these should move to GeometricOptimizers. We could do this first by opening a separate branch in that repo (branched off from GeometricOptimizers#main). We can then delete the functionality from `GeometricMachineLearning` but also open an issue that references the pr in `GeometricOptimizers` and leave the integration as a future task.
8. We can temporaily restructure the test helpers. But this should go hand-in-hand with opening a new issue describing the problem.
9. You can make this change, but we should hence tag michakraus to inform him of this change and ask whether he agrees.

## 7. Verification checklist

|> [!IMPORTANT]
|> For locally running everything, you can access the different Julia versions in ~/.julia/juliaup/.

* [ ] GO v0.2.0 visible in General; `Pkg.add("GeometricOptimizers")` gives 0.2.0.
* [ ] `Project.toml` has no `[sources]`; `julia = "1.10"`.
* [ ] All three Julia 1.10 jobs pass.
* [ ] All three Julia 1.12 jobs pass (currently the only end-to-end signal —
      re-check run `31570166729` once it finishes; if 1.12 is *also* failing for
      a reason not listed above, this plan is incomplete and needs revisiting
      before any of it is implemented) - also confer the comment above.
* [ ] **R8 gone:** 1.12 × ubuntu completes in roughly the ~30 min it takes on
      `main`, not merely "eventually". A job that passes after several hours has
      not fixed R8, it has only outlasted it.
* [ ] One full 1.12 suite has been observed to finish end to end, locally or in
      CI, so that testsets *after* the symplectic-autoencoder one have actually
      been exercised at least once. The local investigation did not get that
      far.
* [ ] `test_accuracy(10, 6; n_epochs = 1)` compiles in seconds from a cold
      session (the minimal R8 regression check).
* [ ] Julia 1.13 passes, or is explicitly marked experimental with a linked
      upstream issue. The CI policy change is applied; the upstream issue is
      still outstanding.
* [x] Documenter doctests are excluded from `test/runtests.jl`; former examples
      have unit coverage under `test/`, with duplicate cases integrated into
      existing domain test files.
* [ ] Documentation workflow green (`@docs` + every tutorial executes; the
      duplicate bibliography keys have been removed).
* [ ] PDF workflow green.
* [ ] `docs/Makefile`'s `test_docs` target passes.
* [ ] Issue #234 closed by a PR that *deletes* the `b16267ea` bridge methods
      rather than adding more.
* [ ] `git status` clean; no dev artifacts in the diff.
