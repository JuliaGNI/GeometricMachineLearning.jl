# Workplan: fixing CI on PR #230 and closing issue #234

Status 2026-08-13, branch `use-geometric-optimizers`, verified against the
working tree on that date.

**Two blockers remain: R1 (dependency resolution) and R8 (compile time).**
Everything else is resolved or deliberately deferred out of #230.

---

## 1. Status

### Blockers

| ID | Problem | Fix |
| --- | --- | --- |
| **R1** | `GeometricOptimizers = "0.2"` is not in General (local registry carries only `0.1.0`). `Project.toml` works around it with `[sources]`, which Pkg < 1.11 ignores, so all three Julia 1.10 jobs die at `Pkg.instantiate`. `[sources]` also makes GML itself unregistrable. | Phase 0 / Phase 1 |
| **R8** | Inference spins in method-table intersection when `GeometricOptimizers.update!` is reached through GML's optimizer tree. 1.12 jobs ran > 1 h 15 min vs ~25–48 min on `main`; run `31570166729` never left `julia-runtest`. Produces no error — a hung job, not a red one. | Phase 4 |

### Resolved

| ID | Problem | Resolution |
| --- | --- | --- |
| R2 | `src/data_loader/batch.jl` doctests hardcode `shuffle` output; the RNG stream changed in 1.13, and the PR had re-enabled the doctest testset in `test/runtests.jl`, aborting the suite before any real test ran. | 2026-08-12: doctests removed from `test/runtests.jl`; the docs workflow keeps ownership of doctest validation, and the examples' behaviour is covered by structural assertions in existing `test/` files. Preferred over reseeding (breaks on the next stream change) and `doctestfilters` (hides real regressions). |
| R3 | `GenericLinearAlgebra v0.4.0` redefines `LinearAlgebra.eigencopy_oftype(::UpperHessenberg, S)`; on Julia ≥ 1.13 a hard precompile error. Chain: `GeometricIntegrators → RungeKutta → GenericLinearAlgebra`. | 2026-08-13: `GeometricIntegrators v0.18.0` drops `GenericLinearAlgebra`, moves to `RungeKutta v0.6`, is registered, and precompiles + loads on 1.13 in a minimal environment. Only full-matrix confirmation outstanding (Phase 3). |
| R4 | Duplicate `Kraus:2020:GeometricIntegrators` and `greydanus2019hamiltonian` BibTeX keys. | Removed; `docs/src/GeometricMachineLearning.bib` verified duplicate-free. (`docs/build/` holds a stale copy — untracked artifact.) |
| R5 | `@docs` blocks referencing symbols that moved to GO or no longer exist. | Applied; one call site left (Phase 2). |
| R6 | Tutorials calling GO v0.1 optimizer constructors removed in v0.2. | Applied; no stale `GradientOptimizer` / `MomentumOptimizer` / `AdamOptimizer(η)` / `BFGSOptimizer` call sites remain under `docs/src/`. |
| — | A long job was indistinguishable from a hung one. | `test/runtests.jl` emits seven `@info` markers, one per testset group. Per §6.9, michakraus still needs to be told. |
| — | Repo hygiene | PR-only workplan and inspection script removed; BFGS test renamed to `test/optimizers/gradient_optimizer.jl`; tracked MNIST PDFs and stray `.jld2` deleted; generated-artifact patterns added to `.gitignore`. |

### Deferred out of #230

**R7 / issue #234** — GML's `StiefelManifold`/`GrassmannManifold` subtype
`GeometricMachineLearning.Manifold`, distinct from
`GeometricOptimizers.Manifold`, so GO's generic `geodesic`/`cayley` never
dispatch. Commit `b16267ea` worked around this by re-implementing GO's pipeline
four times (`geodesic`/`cayley` × `Stiefel`/`Grassmann`), with the same
duplication on the Lie-algebra-horizontal types and the
`_copyto!`/`_add!`/`_rac!`/`_square!`/`_div!` family — ~30 bridge methods, plus
near-verbatim copies of GO's `src/manifolds/*` and `src/arrays/*`. Confirmed
non-critical for #230. Options in Phase 5.

---

## 2. R8 in detail

Two local investigations agree on the mechanism and contradict each other
nowhere.

**Mechanism.** Inference of a call reaching `GeometricOptimizers.update!`
through `_tree_optim_step!`/`_leaf_optim_step!` spins in `ml_matches` →
`jl_type_intersection_env_s` → `ijl_type_union` → `subtype_unionall`. `SIGINFO`
samples land in `subtype_unionall`, `subtype_tuple_varargs`, `jl_type_union`
and method intersection — signature shape and method-table enumeration, not a
large function body. Leading hypothesis: GO's union aliases
(`OptimizerSolution`, `GradientArrayOrNamedTuple`,
`GlobalSectionSingleOrNamedTuple`) used as type-parameter *bounds* on every
optimizer cache and state struct, especially those with `NamedTuple` and
`Vararg` members. Read off backtraces, **not confirmed by profiling** — that is
Phase 4 step 1.

**Not an assertion failure or a deadlock.** On Julia 1.12.6 with a writable
temporary depot: dependency setup (incl. GO), doctests and PSD tests all pass;
the first optimizer epoch takes ~13 s of compilation and later epochs ~7 ms,
with 100 epochs reaching the expected accuracy. Instrumented
`test_symplecticity(10, 6)` completes in ~18 s (first Jacobian ~7.6 s, 10
epochs ~8.7 s, second Jacobian ~0.3 s). Narrowing `N::Integer`/`n::Integer` to
`Int` does not help.

**A user-facing bug, not just a CI one.** The same SAE workload is fast at top
level (~13–18 s cold, ms warm, including under `--check-bounds=yes -g1`); the
pathology appears only when the optimizer is compiled *through* a function — so
every user who wraps training in a function pays it. Restructuring the tests
would hide it.

**Not version-specific.** 1.12 is merely the only matrix entry that gets far
enough; 1.10 dies at R1, 1.13 died at R2.

**Independent of R2.** The doctest testset passed on 1.12 before the stall, and
stack samples show a `@safetestset` module already evaluating, which happens
only after that testset returns.

**Never observed end to end.** Neither investigation finished a full 1.12
suite, so every testset after `symplectic_autoencoder` is unexercised.

---

## 3. Plan

### Phase 0 — Release GeometricOptimizers v0.2.0 (upstream)

Owner: michakraus / JuliaGNI release rights.

1. In `../GeometricOptimizers` on `main`: confirm `version = "0.2.0"`, confirm
   CI green, tag and push.
2. Register in General (`@JuliaRegistrator register` on the release commit);
   verify it lands in `General/G/GeometricOptimizers/Versions.toml`.
3. Sanity-check GO's `[compat]`, in particular `SimpleSolvers = "0.10"` and
   `julia = "1.10"` — GML inherits those constraints.

Blocked on release rights and on §6.6 (whether Phase 4 step 2 rides along in
0.2.0 or follows as 0.2.1).

### Phase 1 — Make GML resolvable

**Decided: interim variant** (§6.1) — GO 0.2.0 is not in General, so the target
form is unreachable. In `Project.toml`:

* Keep `[sources]` and `GeometricOptimizers = "0.2"`.
* `julia = "1.9"` → `"1.11"` (1.9 was never satisfiable with this dependency
  set; `[sources]` needs Pkg ≥ 1.11).
* Drop `"1.10"` from the matrix in `.github/workflows/CI.yml`.
* Open a GML issue: 1.10 support must be restored.
* Add compat bounds for any new direct dependency lacking one (check
  `ParameterHandling`, `SimpleSolvers` if now direct).
* Say in the PR description that this is the interim variant.

After Phase 0 this reverts to the target form — delete `[sources]`, set
`julia = "1.10"` (GO's floor), restore 1.10 to the matrix, close the issue —
which also unblocks GML's registration. The variants are mutually exclusive.
`docs/Project.toml` has no `[sources]` and needs no change either way.

### Phase 2 — Documentation and PDF builds

**Done.** BFGS is gone from GML, so `docs/src/optimizers/bfgs_optimizer.md` was
deleted and `docs/make.jl` no longer references it in the HTML nav or the LaTeX
page list. `optimizer_methods.md` is now prose linking out to GO, with `@docs`
only for GML-owned symbols. `Optimizer` (`src/utils.jl:251`),
`AdamOptimizerWithDecay` (`:208`) and `optimization_step!` (`:390`) have
docstrings. Every R6 call site moved to the v0.2 API, learning rate out of the
method constructor and into `Optimizer(...; step_size = …)`:

```julia
o = Optimizer(nn, AdamOptimizer(1e-1))              # before
o = Optimizer(Adam(Float64), nn; step_size = 1e-1)  # after
```

Per §6.2 the old signatures were **not** restored — neither network-first
`Optimizer(nn, method)` nor type-first `AdamOptimizerWithDecay(n_epochs,
T::Type)`. The break belongs in the release notes / changelog.

**Outstanding: one broken cross-reference.**
`docs/src/optimizers/manifold_related/retractions.md:345` links
``[`GeometricMachineLearning.𝔄`](@ref)``, but `𝔄` exists only as
`GeometricOptimizers.𝔄` (GML calls it at `src/manifolds/stiefel_manifold.jl:272`
and `src/manifolds/grassmann_manifold.jl:207`), so the `@ref` cannot resolve.
Replace with prose plus a link to GO's docs, or use `DocumenterInterLinks`.

Do **not** add `GeometricOptimizers` to `modules=[...]` in `docs/make.jl`:
Documenter's default `checkdocs = :all` would demand a page for every GO
docstring and produce hundreds of new errors.

### Phase 3 — Julia 1.13 and nightly

`CI.yml` already marks all three `^1.13.0-0` and all three nightly jobs
`experimental: true`, so the §6.4 policy is in place. Remaining:

* Run the full matrix and confirm it resolves `GeometricIntegrators v0.18.0`;
  if any workflow selects an older version, refresh the generated environment
  artifacts first.
* If 1.13 is then green, remove its experimental status.
* Open the GML issue for restoring required 1.13 support — per §6.4, first
  check `RungeKutta`/`GeometricIntegrators` open issues for an existing
  `GenericLinearAlgebra` report to reference.

### Phase 4 — R8: make the optimizer path compile in reasonable time (blocking)

1. **Localise it.** `@snoopi_deep` (SnoopCompileCore) on
   `test_accuracy(10, 6; n_epochs = 1)`, or `--trace-compile=stderr` for the
   last signature compiled before the spin. Decides whether the GO aliases are
   the cause or merely plausible.
2. **Fix upstream in GO** if confirmed: stop using `OptimizerSolution` /
   `GradientArrayOrNamedTuple` / `GlobalSectionSingleOrNamedTuple` as
   type-parameter *bounds* on the cache and state structs
   (`optimizer_solution.jl:4-26` and the six struct definitions); enforce the
   invariant in inner constructors instead. Behaviour unchanged.
3. **Reduce GML's dispatch fan-out.** *Partly done*: the `adapted isa …` branch
   in `_leaf_optim_step!` is now concrete dispatch via `_go_update_leaf!`
   (`src/utils.jl:316-331`), with overloads for `Adam`, `MomentumMethod` and
   the fallback `OptimizerMethod`. Two `state isa` branches remain in the same
   function (`:355`, `:360`, for `AdamState`/`MomentumState`) — convert them
   too if step 1 implicates them.
4. **Do not "fix" it by restructuring the tests**, except as a temporary
   unblock with a linked issue (§6.8): that leaves the latency in place for
   every user who wraps training in a function.

Success criterion: 1.12 × ubuntu back to the ~30 min it takes on `main`, and
`test_accuracy(10, 6; n_epochs = 1)` compiling in seconds.

**Coupled to §4:** if the `src/utils.jl` traversal is replaced by GO's native
NamedTuple handling, step 3 disappears with it — which may make §4 the real fix
rather than a cleanup.

The local step-3 mitigation is **not runtime-validated** — the local Julia depot
is read-only and network access unavailable. Passing offline checks: bibliography
uniqueness, Julia parsing, stale-reference scans, `git diff --check`.

### Phase 5 — Issue #234 (separate PR, does not block #230)

**Option A — alias GML's abstract type to GO's. Decided (§6.5).**

```julia
const Manifold = GeometricOptimizers.Manifold
```

GML keeps its concrete `StiefelManifold`/`GrassmannManifold` types and their
GML-specific methods, but they now subtype GO's abstract `Manifold`, so GO's
generic `geodesic`/`cayley` dispatch directly and all four `b16267ea` bridge
methods can be deleted with no change to GO.

⚠️ Part of the change, not a follow-up: `src/manifolds/abstract_manifold.jl` is
a near-verbatim copy of GO's, so after aliasing GML's generic methods
(`Base.rand`, `size`, `getindex`, `copy`, the `similar`/`fill!` error methods,
broadcasting, `_round`) would have *identical* signatures to GO's and silently
overwrite them — on Julia ≥ 1.13 a hard precompilation error, i.e. R3
re-created from inside GML. So delete GML's copies, keeping only what genuinely
differs (e.g. `rand` dispatching on GML's `networkbackend`/device types, which
has a distinct signature). Same for `AbstractLieAlgHorMatrix`.

**Option B — a Holy trait in GO**, letting any external hierarchy opt in:

!!!info comment by benedit-96
    I think this is not a good option. I left it in there for now, but could probably be removed after quickly checking with michakraus.

```julia
abstract type ManifoldTrait end
struct IsHomogeneousSpace <: ManifoldTrait end
struct NotAManifold      <: ManifoldTrait end
ManifoldTrait(::Type)                = NotAManifold()
ManifoldTrait(::Type{<:Manifold})    = IsHomogeneousSpace()

geodesic(Y::AbstractMatrix{T}, Δ::AbstractMatrix{T}) where {T} =
    geodesic(ManifoldTrait(typeof(Y)), Y, Δ)
```

with the generic body factored into a shared `_retract(retraction, Y, Δ)`
calling `apply_section(λY, …)` instead of `λY * …`. Only if GML must keep an
independent hierarchy: more machinery for the same result, and it turns
`geodesic(::AbstractMatrix, ::AbstractMatrix)` from a `MethodError` into a
custom error.

**Option C — full type unification (long-term, own issue per §6.5).** GML stops
defining manifold, Lie-algebra-horizontal and structured-matrix types and
re-exports GO's, deleting `src/manifolds/*` and much of `src/arrays/*`. The
right end state, but a large refactor touching layers, AD rules,
`networkbackend`, `Ω`, GPU kernels operating on `.A`, and `rand` on GML device
types — and some `Base` methods on GO types would become type piracy.

---

## 4. Move optimizer machinery to GO

Per §6.7 **all** of this moves to GO — not optional cleanup:

* `AdamOptimizerWithDecay` (`src/utils.jl:208`) — **done upstream**, see below;
* the traversal: `_make_optimizer_cache` (`:231`), `_make_optimizer_state`
  (`:241`), `_tree_optim_step!` (`:377`), `_leaf_optim_step!` (`:334`);
* the bespoke `GMLEuclideanState` (`:199`) holding `m₁`/`m₂` by hand.

GO v0.2 already supports NamedTuple-valued solutions natively
(`GradientArrayOrNamedTuple`, `OptimizerSolution`, `GlobalSection(::NamedTuple)`,
`ParameterHandling.flatten` for `Manifold`) and already has
`AdamCache`/`AdamState`/`MomentumCache`/`MomentumState`, so most of these ~262
lines are replaceable by GO's own machinery.

Route: branch `GeometricOptimizers`, move the functionality there, delete it from
GML, open a GML issue referencing the GO PR with integration left as a future
task. Normally a follow-up — but if Phase 4 step 1 confirms the traversal is the
R8 trigger, doing it now is the *smallest correct fix*.

### `AdamOptimizerWithDecay` → [GeometricOptimizers#33][go33] (draft)

[go33]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/33

The first item is upstream. GML's method turned out to need no port: GO's
`DecayingStatic` line search already implements its schedule factor for factor
(γ = exp(log(η₂/η₁)/n), step η₁γᵗ). What GO lacked was the *name* — 0.2.0 split
the step size out of the `OptimizerMethod`s, and the bundling of Adam's ρ₁, ρ₂,
δ with η₁, η₂, n_epochs went with it.

GO#33 restores the name as a convenience pairing returning the
`(algorithm, linesearch)` NamedTuple, so GML **deletes** `src/utils.jl:208-229`
and rewrites call sites as:

```julia
Optimizer(x, problem; AdamOptimizerWithDecay(n_epochs)...)
```

Also in GO#33: `test/adam_optimizer_with_decay.jl` (33 assertions, including
`step_size(ls, t) ≈ η₁·γ_gmlᵗ` — the claim that licenses the deletion) and a
"Two unrelated decays" docs section separating this *learning-rate* decay from
`AdamWithEuclideanDecay`'s *weight* decay.

Stacked and **draft**: `DecayingStatic` lives only on
`docs-linesearch-on-manifolds` and the weight-decay docs only on
`manifold-adamw`, so §6.7's "branch GO off `main`" was not possible for this
item. Base is `manifold-adamw` ([GO#29][go29]), which must merge first; `main`
and `docs-linesearch-on-manifolds` have been merged into it and pushed. The
GML-side deletion therefore lands after GO#29 and GO#33, not in this PR.

That merge forced one semantic choice — which line search
`AdamWithEuclideanDecay` defaults to — written up for confirmation in
[GO#34][go34]. It does not block anything here.

[go29]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/29
[go34]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/34

---

## 5. Order of execution

```
Phase 4 (R8)  ── BLOCKING, start first; needs no dependency resolution to
                 reproduce. Step 2 should land in GO before Phase 0 tags.

Phase 1 (interim Project.toml + 1.10 issue) ──→ 1.10 removed from matrix
Phase 0 (upstream release)                  ──→ later reverts Phase 1 to target form
Phase 2 (retractions.md @ref)               ──→ Documentation + PDF green
Phase 3 (1.13 confirmation)                 ──→ 1.13/nightly de-experimentalised (comment by benedict-96: nightly can stay experimentalised as is the case in all other JuliaGNI repos)
Phase 5 (#234)                              ──→ separate PR
§4 (move to GO)                             ──→ AdamOptimizerWithDecay: GO#33
                                                (draft, stacked on GO#29);
                                                traversal: separate GO branch
                                                + GML issue
```

Phases 0–3 can all land and *every* job will still be red or hanging until
Phase 4 is done, since R8 blocks the only jobs that reach the test suite. Phase
4 is therefore first in practice, even though Phase 0 is the formal critical
path.

---

## 6. Decisions (questions by claude, answers by benedict-96)

1. **Tag and register GO v0.2.0 now; if not, is dropping Julia 1.10 acceptable
   as an interim?** → Dropping 1.10 is acceptable, provided an issue records
   that it must be restored. Fix the other CI jobs first.
2. **Restore backwards-compatible `Optimizer(nn, method)` and
   `AdamOptimizerWithDecay(n_epochs, T::Type)`, or update every call site?** →
   Update every call site; optimizer functionality belongs in GO, not GML.
3. **Delete `bfgs_optimizer.md` or rewrite it around GO's `_BFGS()`?** → Code
   replicated from GO must be removed from GML. Where optimizer functionality
   exists in GML but not GO, pause and add a separate workplan file.
4. **Must `^1.13.0-0` stay required while GenericLinearAlgebra is broken?** →
   No; make it non-required, open a GML issue for restoring it, and first check
   `RungeKutta`/`GeometricIntegrators` for an existing `GenericLinearAlgebra`
   report to reference.
5. **Option A now for #234, Option C as follow-up?** → Yes — and open the issue
   for C.
6. **R8 GO struct-signature change before v0.2.0, or as v0.2.1?** → If it does
   not change the API, wait for v0.2.1; if it does, open an issue at the
   appropriate spot and tag michakraus.
7. **Does "optimizer functionality belongs in GO" extend to
   `AdamOptimizerWithDecay` and the §4 traversal?** → Yes, all of it. Branch GO
   off `main`, move it there, delete from GML, and open a GML issue referencing
   the GO PR, integration left as a future task.
8. **If R8 needs an upstream Julia fix, may the test helpers be restructured
   temporarily?** → Yes, hand-in-hand with a new issue describing the problem.
9. **Per-testset progress markers in `runtests.jl`?** → Yes, but tag michakraus
   to inform him and ask whether he agrees.

---

## 7. Verification checklist

> [!IMPORTANT]
> The different Julia versions are available locally under `~/.julia/juliaup/`.

**R1 / resolution**
* [ ] Interim applied: `julia = "1.11"`, 1.10 removed from `CI.yml`, tracking
      issue opened.
* [ ] GO v0.2.0 visible in General; `Pkg.add("GeometricOptimizers")` gives 0.2.0.
* [ ] Target form applied: no `[sources]`, `julia = "1.10"`, 1.10 back in the
      matrix, all three 1.10 jobs pass.

**R8**
* [ ] `test_accuracy(10, 6; n_epochs = 1)` compiles in seconds from a cold
      session (minimal regression check).
* [ ] 1.12 × ubuntu completes in roughly the ~30 min it takes on `main` — a job
      that passes after several hours has outlasted R8, not fixed it.
* [ ] All three 1.12 jobs pass. Re-check run `31570166729` once it finishes; if
      1.12 also fails for a reason not listed here, this plan is incomplete and
      needs revisiting before more of it is implemented.
* [ ] One full 1.12 suite observed end to end, so the testsets after
      `symplectic_autoencoder` are exercised at least once.

**Docs**
* [x] Doctests excluded from `test/runtests.jl`; former examples covered under
      `test/`.
* [x] `docs/src/GeometricMachineLearning.bib` free of duplicate keys.
* [ ] `retractions.md:345` no longer `@ref`s `GeometricMachineLearning.𝔄`.
* [ ] Documentation workflow green (every `@docs` entry and tutorial executes).
* [ ] PDF workflow green.
* [ ] `docs/Makefile`'s `test_docs` target passes.

**1.13**
* [x] 1.13 and nightly marked `experimental: true` in `CI.yml`.
* [ ] Full matrix confirmed to resolve `GeometricIntegrators v0.18.0`.
* [ ] 1.13 green and de-experimentalised, or the linked upstream issue opened.

**Follow-ups**
* [ ] michakraus informed about the `runtests.jl` progress markers.
* [x] GO branch opened for the §4 move: [GO#33][go33] (draft, base `manifold-adamw`),
      covering `AdamOptimizerWithDecay`.
* [ ] [GO#34][go34] answered: `AdamWithEuclideanDecay`'s default line search confirmed.
* [ ] GO#29 merged, then GO#33 undrafted and merged.
* [ ] `AdamOptimizerWithDecay` deleted from `src/utils.jl` and call sites moved to
      `Optimizer(x, problem; AdamOptimizerWithDecay(n)...)`.
* [ ] GO branch opened for the §4 traversal (`_make_optimizer_cache`,
      `_make_optimizer_state`, `_tree_optim_step!`, `_leaf_optim_step!`,
      `GMLEuclideanState`); GML issue opened referencing both GO PRs.
* [ ] Issue opened for #234 Option C.
* [ ] Issue #234 closed by a PR that *deletes* the `b16267ea` bridge methods
      rather than adding more.
* [ ] `git status` clean; no dev artifacts in the diff.
</content>
