# Workplan: fixing CI on PR #230 and closing issue #234

Status 2026-08-15, branch `use-geometric-optimizers`, verified against the
working tree, the General registry, the GitHub APIs and live runs on Julia
1.10.11 and 1.12.6 on that date.

**Every remaining blocker is an unreleased upstream package; no GML-side work is
left.** R1 is gone — GeometricOptimizers v0.2.0 is registered. R9 (new, and not
in earlier revisions of this plan) needs SimpleSolvers 0.12.1,
GeometricIntegratorsBase 0.6.3 and GeometricIntegrators 0.18.2 released before
anything resolves. R8 is *solved* — cause located by profiling, fix verified end
to end — but the fix is in GO and needs a 0.2.1. All four are Phase 0.

> [!IMPORTANT]
> Local verification is possible now. `~/.julia` is writable, the registry is
> reachable, and `~/.julia/juliaup/` carries 1.10.11, 1.12.6 and 1.13.0-rc2.
> Earlier revisions of this plan recorded a read-only depot with no network and
> marked several claims "not runtime-validated"; that caveat is withdrawn.

---

## 1. Status

### Blockers

| ID | Problem | Fix |
| --- | --- | --- |
| **R9** | GO 0.2.0 requires `SimpleSolvers = "0.12"`; registered `GeometricIntegrators 0.18.1` requires `"0.11"`. No overlap, and GI is in GML's test target and in `docs/Project.toml`, so **every** job dies at resolution in ~40 s — all twelve CI jobs plus Documentation and PDF, the latter two on `make test_docs` with `Unsatisfiable requirements detected for package SimpleSolvers`. | Phase 0 |

### Resolved

| ID | Problem | Resolution |
| --- | --- | --- |
| R1 | `GeometricOptimizers = "0.2"` was not in General, and the `[sources]` workaround is ignored by Pkg < 1.11, killing the three Julia 1.10 jobs and making GML unregistrable. | 2026-08-15: GO v0.2.0 tagged (04:19Z) and registered, with `julia = "1.10.0 - 1"`. `[sources]` deleted, `julia` raised `"1.9"` → `"1.10"`, `GeometricIntegrators = "0.18.2"` bound added. 1.10 stays in the matrix; the interim "drop 1.10" variant was never needed. |
| R2 | `src/data_loader/batch.jl` doctests hardcode `shuffle` output; the RNG stream changed in 1.13, and the re-enabled doctest testset aborted the suite before any real test ran. | 2026-08-12: doctests removed from `test/runtests.jl`; the docs workflow keeps ownership of doctest validation, and the examples are covered by structural assertions under `test/`. |
| R3 | `GenericLinearAlgebra v0.4.0` redefines `LinearAlgebra.eigencopy_oftype(::UpperHessenberg, S)`; on Julia ≥ 1.13 a hard precompile error. Chain: `GeometricIntegrators → RungeKutta → GenericLinearAlgebra`. | Confirmed on a full resolve: with GI 0.18.2 and RungeKutta 0.6.1, `GenericLinearAlgebra` is absent from the manifest entirely. |
| R4 | Duplicate `Kraus:2020:GeometricIntegrators` and `greydanus2019hamiltonian` BibTeX keys. | Removed; `docs/src/GeometricMachineLearning.bib` verified duplicate-free. |
| R5 | `@docs` blocks and `@ref`s pointing at symbols that moved to GO or no longer exist. | Applied. Earlier revisions of this plan said one call site was left; an audit of every `@ref` target against every `@docs` entry found **seven**. See §4/Phase 2. |
| R6 | Tutorials calling GO v0.1 optimizer constructors removed in v0.2. | Applied; no stale `GradientOptimizer` / `MomentumOptimizer` / `AdamOptimizer(η)` / `BFGSOptimizer` call sites remain under `docs/src/`. |
| **R8** | Inference spun in method-table intersection whenever the optimizer path was compiled through a function. 1.12 jobs ran > 1 h 15 min vs ~25–48 min on `main`; run `31570166729` never left `julia-runtest`. A hung job, not a red one. | **Cause located and fix verified — see §2.** GO's six optimizer cache/state structs bound their type parameters by the `OptimizerSolution` / `GradientArrayOrNamedTuple` / `GlobalSectionSingleOrNamedTuple` union aliases. Dropping those bounds takes the repro from *never completing* to 14.3 s cold / 6.9 ms warm. Needs a GO release. |
| — | GML did not precompile on Julia 1.10 at all: `cannot assign a value to imported variable GeometricOptimizers.Manifold`. | Found during Phase 3 verification. `using GeometricOptimizers` was a blanket import, and GO exports ~20 names GML defines itself. Replaced with `import GeometricOptimizers` plus an explicit `using GeometricOptimizers: …` list. See §3 — this also fixed a silent correctness bug. |
| — | A long job was indistinguishable from a hung one. | `test/runtests.jl` emits seven `@info` markers, one per testset group. Per §7.9, michakraus still needs to be told. |
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

Note that §3's bug is the same type split showing up somewhere else, so #234 is
worth more than the tidiness it looks like.

---

## 2. R8 in detail — cause confirmed, fix verified

Earlier revisions of this plan named GO's union aliases as a *leading
hypothesis* read off `SIGINFO` backtraces and explicitly not confirmed by
profiling. That hypothesis was right, down to the construct. What was wrong was
the suspicion that GML's own dispatch fan-out contributed, and the suggestion
that the §5 traversal move might turn out to be the real fix: both were measured
and neither is (see *Not a GML-side fix* below).

**Repro.** `test_accuracy(10, 6; n_epochs = 1)`, i.e.
`test/symplectic_autoencoder_tests.jl:8`, on Julia 1.12.6.

**Mechanism.** Under SIGTERM the stall is always in
`typeinf_ext_toplevel → abstract_call_gf_by_type → find_simple_method_matches →
ml_matches → jl_typemap_intersection_visitor → jl_type_intersection_env_s →
ijl_type_union → subtype_unionall`, having burned > 4 × 10⁹ allocations. So it
is inference doing method-table intersection, exactly as suspected.

**What makes the signature abstract.** Split into four separate one-statement
functions, the four stages total ~14 s cold (0.24 + 0.57 + 0.91 + 12.29) and the
epoch is 6.8 ms warm; put into a single method body the same work never
finishes. The difference is that in one body the type of
`Optimizer(Adam(), sae_nn)` has to be *inferred* rather than taken from a
concrete argument. Inference gives

```julia
Optimizer{Adam{Float64}, CT, ST, typeof(cayley)} where {CT<:NamedTuple{(:L1, …, :L9)},
                                                        ST<:NamedTuple{(:L1, …, :L9)}}
```

— a `UnionAll` with two unconstrained `NamedTuple` parameters — whereas the
runtime type is a fully concrete nine-field `NamedTuple` of `AdamCache`s. Every
subsequent call on that `Optimizer` therefore intersects an abstract signature
against the whole method table, and that is the explosion.

**Where it comes from.** Not from GML's traversal, but one level down, from the
bounds on GO's six cache and state structs in
`src/manifold_optimizers/{adam,momentum,gradient}_optimizer.jl`:

```julia
struct AdamCache{T, MT<:OptimizerSolution{T},
                    VT<:GradientArrayOrNamedTuple{T},
                    ST<:GlobalSectionSingleOrNamedTuple{T}} <: OptimizerCache{T}
```

Those aliases are `Union`s of `UnionAll`s over `Tuple{Vararg{AbstractArray{T}}}`
(`src/optimizer_solution.jl:1-26`).

It is worth being exact about *how* they hurt, because the obvious explanation is
wrong. They do not cost concrete inference: `OptimizerCache(Adam(Float64), ps)`
on a `NamedTuple` of parameters infers to a `UnionAll` with or without them, and
removing them does not make it concrete. GO's outer constructors are written in
the same aliases and are enough to cause that by themselves.

What the bounds add is *coupling*. With them the inferred type is

```julia
AdamCache{T, NamedTuple{(:Y,:W,:b),s1}, NamedTuple{(:Y,:W,:b),s2}, NamedTuple{(:Y,:W,:b),s3}} where
    {T, s1<:Tuple{Vararg{AbstractArray{T}}}, s2<:Tuple{Vararg{AbstractArray{T}}},
        s3<:Tuple{Vararg{GlobalSection{T,AT,λT} where {AT<:AbstractArray{T},
                                                       λT<:Union{Nothing,AbstractArray{T}}}}}}
```

— a single `T` tying all four parameters together under three nested `Vararg`
unions, the last over a three-parameter `UnionAll`. Every method-table
intersection involving such a type re-solves that constraint system in
`subtype_unionall`. Without the bounds the same call infers to the same *shape*
but with the parameters independent and `s3<:Tuple`, which costs nothing to
intersect.

**Verified fix.** Dropping the bounds from all six structs —
`struct AdamCache{T, MT, VT, ST} <: OptimizerCache{T}`, and the same for
`AdamState`, `MomentumCache`, `MomentumState`, `GradientCache`, `GradientState` —
takes the single-body repro from *never completing* to **~14.5 s cold / 6.5 ms
warm**. Measured three times: twice against a throwaway copy of GO 0.2.0 and once
against the real branch. Nothing else changed.

No inner constructors are needed, contrary to what earlier revisions of this plan
assumed: the invariant is already enforced by the outer constructors, whose own
signatures take `x::OptimizerSolution{T}` and
`g::AT where AT<:GradientArrayOrNamedTuple{T}` and build the `GlobalSection`
themselves. That is the same guarantee, checked by dispatch, at no cost to
inference.

**Not a GML-side fix.** Rewriting GML's `_make_optimizer_cache` /
`_make_optimizer_state` from `isa` branches to dispatch was tried and measured:
against released GO it still never completes, and against patched GO it is
14.25 s versus 14.55 s for the existing branchy code — i.e. within noise. The
rewrite was therefore **not** kept, and `_go_update_leaf!` and the two remaining
`state isa` branches need no further work. The traversal is unimplicated.

**A user-facing bug, not just a CI one.** The same workload is fast at top level
and pathological through a function, so every user who wraps training in a
function pays it. Restructuring the tests would have hidden it.

**Not version-specific.** 1.12 was merely the only matrix entry that got far
enough; 1.10 died at R1 and 1.13 at R2.

---

## 3. The Julia 1.10 precompilation failure (and the bug behind it)

Found while checking that the `julia = "1.10"` claim Phase 1 newly makes is
honest — no 1.10 job had ever run.

`src/GeometricMachineLearning.jl:27` had a blanket `using GeometricOptimizers`.
GO exports about twenty names GML defines itself: `Manifold`,
`StiefelManifold`, `GrassmannManifold`, `SkewSymMatrix`, `SymmetricMatrix`,
`LowerTriangular`, `UpperTriangular`, `StiefelLieAlgHorMatrix`,
`GrassmannLieAlgHorMatrix`, `Optimizer`, `AdamOptimizerWithDecay`, `rgrad`, …
Julia 1.12's binding partitions tolerate redefining an imported binding; 1.10
raises `cannot assign a value to imported variable GeometricOptimizers.Manifold`
and GML does not precompile.

The fix is `import GeometricOptimizers` plus an explicit
`using GeometricOptimizers: …` list, extended with the names GML re-exports
(`GradientMethod`, `MomentumMethod`, `Adam`, the three `*State`s, `GlobalSection`,
`global_rep`).

**That exposed a silent correctness bug.** `include("utils.jl")` runs at line 71,
`include("manifolds/abstract_manifold.jl")` at line 142, so `Manifold` in

```julia
_gml_rgrad(x::Manifold, dp) = rgrad(x, dp)
```

resolved to **`GeometricOptimizers.Manifold`**. GML's `StiefelManifold` does not
subtype that (`GML.Manifold === GO.Manifold` is `false`, and GML's is
`<: AbstractMatrix`), so the method never matched a GML manifold: the
`_gml_rgrad(x, dp) = dp` fallback caught it and the Riemannian projection was
skipped, passing the raw Euclidean gradient through.

The optimizer machinery was therefore moved out of `utils.jl` (lines 177-400)
into a new `src/optimizers/optimizer.jl`, included after the manifolds. The move
is byte-for-byte — `diff` against the old `utils.jl` section is empty — so the
whole change is the relocation plus the include. `_gml_rgrad` now dispatches to
`src/optimizers/optimizer.jl:16` for a `StiefelManifold`, as it was meant to.

---

## 4. Plan

### Phase 0 — Release the three upstream packages (blocking, upstream)

Owner: michakraus / JuliaGNI release rights. This is now the entire critical
path.

The local checkouts already carry the fix and are what local verification uses:

| package | local path | version | requires |
| --- | --- | --- | --- |
| SimpleSolvers | `/Users/mkraus/Julia/SimpleSolvers` | 0.12.1 | — |
| GeometricIntegratorsBase | `/Users/mkraus/Julia/GeometricIntegratorsBase` | 0.6.3 | SS 0.12.1 |
| GeometricIntegrators | `/Users/mkraus/Datashare/Julia/GeometricIntegrators` | 0.18.2 | SS 0.12.1, GIB 0.6.3 |

All three have uncommitted changes. Tag and register them in dependency order.
GI's CompatHelper PR [#239][gi239] does the same bump on `main` and has only its
*experimental* 1.13/nightly checks failing, so it is an alternative route to
0.18.2.

[gi239]: https://github.com/JuliaGNI/GeometricIntegrators.jl/pull/239

Alongside: **GO 0.2.1 with the §2 struct-bound fix**, opened as a PR from
`inference-friendly-cache-parameters`. Per §7.6 this belongs in a patch release
because it does not change the API — the bounds come off the type parameters,
not off what the constructors accept, and the outer constructors still reject
exactly what they always did. Until it is out, CI cannot pass even once
resolution works, because R8 returns.

### Phase 1 — Make GML resolvable

**Done, in target form.** GO 0.2.0 is in General, so the interim variant of this
phase (keep `[sources]`, drop 1.10 from the matrix, open a tracking issue) was
skipped entirely. In `Project.toml`:

* `[sources]` deleted — GML is registrable again;
* `julia = "1.9"` → `"1.10"`, GO's floor (1.9 was never satisfiable with this
  dependency set);
* `GeometricIntegrators = "0.18.2"` added to `[compat]`. It is a test-only
  dependency with no bound, so Pkg was free to pick an older GI whose
  SimpleSolvers requirement conflicts with GO's, turning R9 into a confusing
  resolver tree rather than a clear "no such version yet".

`.github/workflows/CI.yml` is unchanged: 1.10 stays in the matrix.
`docs/Project.toml` needs no change.

### Phase 2 — Documentation and PDF builds

**Done.** BFGS is gone from GML, so `docs/src/optimizers/bfgs_optimizer.md` was
deleted and `docs/make.jl` no longer references it in the HTML nav or the LaTeX
page list. `optimizer_methods.md` is now prose linking out to GO, with `@docs`
only for GML-owned symbols. `Optimizer`, `AdamOptimizerWithDecay` and
`optimization_step!` have docstrings. Every R6 call site moved to the v0.2 API,
learning rate out of the method constructor and into `Optimizer(...; step_size = …)`:

```julia
o = Optimizer(nn, AdamOptimizer(1e-1))              # before
o = Optimizer(Adam(Float64), nn; step_size = 1e-1)  # after
```

Per §7.2 the old signatures were **not** restored. The break belongs in the
release notes / changelog.

**Eight unresolvable cross-references, not one.** Neither workflow can be run
until Phase 0 — both die at resolution long before Documenter starts — so the
`@ref`s were audited statically instead. There is no `@autodocs` block anywhere
in `docs/`, so an `@ref` resolves only if some `@docs` entry documents that
binding; extracting all 61 `@ref` targets and all 142 `@docs` entries and
matching them modulo the `GeometricMachineLearning.` prefix is therefore a sound
check — provided the extraction also accepts `@docs` blocks carrying attributes,
of which `docs/src/introduction.md:21` has one (` ```@docs; canonical = false `).
Seven targets have **zero** `@docs` entries:

| target | status | fix |
| --- | --- | --- |
| `GeometricMachineLearning.𝔄` | exists only as `GeometricOptimizers.𝔄` | prose + link to GO's docs |
| `cayley(::StiefelLieAlgHorMatrix)`, `cayley(::StiefelManifold, ::AbstractMatrix)` | GML's bridge methods from `b16267ea`, no docstring anywhere | plain code spans |
| `Adam`, `MomentumMethod` | GO-owned; `@ref`s **introduced by this PR** (absent on `main`) | plain code spans |
| `cayley`, `update!` | GO- and AbstractNeuralNetworks-owned; their GML `@docs` entries went with the BFGS/optimizer removal | plain code spans |
| `AdamOptimizerWithDecay` | still GML-owned with a docstring; `main` documented it as a bare `@docs` entry in `optimizer_methods.md`, which the rewrite of that page into prose dropped, dangling two tutorial `@ref`s | `@docs` entry restored |

Everything else checks out and was left alone:

* `geodesic` — `riemannian_manifolds.md` still carries
  `geodesic(::Manifold{T}, ::AbstractMatrix{T}) where T`;
* `GradientMethod` — documented by the attributed block in `introduction.md`,
  which is also the page whose prose calls it out as the example of a
  cross-reference, so de-referencing it would have contradicted the surrounding
  text;
* `global_section`, `iterate`, `mat_tensor_mul`, `tensor_mat_mul`, `Ω`, `rgrad` —
  documented by *signature* rather than by bare name, which is why a naive
  string match reports them as unmatched.

The last group predates this PR: `main` carries the identical `@ref`s and its
Documentation workflow was green until it broke on R4 (run `29832781863`,
2026-07-21, `Duplicate BibTeX entry key`), which this branch already fixes.
Re-running the audit afterwards reports no unresolvable target.

Do **not** add `GeometricOptimizers` to `modules=[...]` in `docs/make.jl`:
Documenter's default `checkdocs = :all` would demand a page for every GO
docstring.

The static audit is not a substitute for the real build: it catches missing
`@docs` entries but says nothing about executable blocks, doctests or tutorials.
Both workflows stay unverified until Phase 0.

### Phase 3 — Julia 1.10, 1.13 and nightly

`CI.yml` already marks all three `^1.13.0-0` and all three nightly jobs
`experimental: true`, so the §7.4 policy is in place; per §6 nightly stays
experimental permanently, as in the other JuliaGNI repos. Remaining:

* 1.10: GML now precompiles and loads (§3). The test suite has not been run
  there yet.
* Confirm on the real matrix that `GeometricIntegrators 0.18.2` is selected. In
  the local resolve it is, and `GenericLinearAlgebra` is absent from the
  manifest, which is R3 confirmed rather than assumed.
* If 1.13 is then green, remove its experimental status.
* Open the GML issue for restoring required 1.13 support — per §7.4, first check
  `RungeKutta`/`GeometricIntegrators` open issues for an existing
  `GenericLinearAlgebra` report to reference.

### Phase 4 — R8

Superseded by §2: located, fixed and verified. The remaining action is the GO
0.2.1 release in Phase 0. Success criterion met locally —
`test_accuracy(10, 6; n_epochs = 1)` compiles in ~14 s from cold instead of
never finishing.

### Phase 5 — Issue #234 (separate PR, does not block #230)

**Option A — alias GML's abstract type to GO's. Decided (§7.5).**

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
overwrite them — on Julia ≥ 1.13 a hard precompilation error, i.e. R3 re-created
from inside GML. So delete GML's copies, keeping only what genuinely differs
(e.g. `rand` dispatching on GML's `networkbackend`/device types, which has a
distinct signature). Same for `AbstractLieAlgHorMatrix`.

Note that §3's `_gml_rgrad` bug is a second instance of the same split, so
Option A retires a class of bug and not just the bridge methods.

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

**Option C — full type unification (long-term, own issue per §7.5).** GML stops
defining manifold, Lie-algebra-horizontal and structured-matrix types and
re-exports GO's, deleting `src/manifolds/*` and much of `src/arrays/*`. The
right end state, but a large refactor touching layers, AD rules,
`networkbackend`, `Ω`, GPU kernels operating on `.A`, and `rand` on GML device
types — and some `Base` methods on GO types would become type piracy.

---

## 5. Move optimizer machinery to GO

Per §7.7 **all** of this moves to GO — not optional cleanup:

* `AdamOptimizerWithDecay` (`src/optimizers/optimizer.jl:35`) — **done upstream**,
  see below;
* the traversal: `_make_optimizer_cache`, `_make_optimizer_state`,
  `_tree_optim_step!`, `_leaf_optim_step!`;
* the bespoke `GMLEuclideanState` holding `m₁`/`m₂` by hand.

GO v0.2 already supports NamedTuple-valued solutions natively
(`GradientArrayOrNamedTuple`, `OptimizerSolution`, `GlobalSection(::NamedTuple)`,
`ParameterHandling.flatten` for `Manifold`) and already has
`AdamCache`/`AdamState`/`MomentumCache`/`MomentumState`, so most of these ~224
lines are replaceable by GO's own machinery.

Route: branch `GeometricOptimizers`, move the functionality there, delete it from
GML, open a GML issue referencing the GO PR with integration left as a future
task. Earlier revisions flagged this as possibly the *smallest correct fix* for
R8 if the traversal turned out to be the trigger; §2 shows it is not, so this is
a follow-up and nothing more.

### `AdamOptimizerWithDecay` → [GeometricOptimizers#33][go33] — merged

[go33]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/33
[go29]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/29
[go34]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/34

GO#33 has merged (as have GO#29 and GO#35), and GO 0.2.0 ships
`AdamOptimizerWithDecay`. GML's method needed no port: GO's `DecayingStatic`
line search already implements its schedule factor for factor
(γ = exp(log(η₂/η₁)/n), step η₁γᵗ), asserted against a live solve in GO's
`test/adam_optimizer_with_decay.jl`.

**The GML-side deletion is nevertheless blocked, on the §5 traversal move.** GO's

```julia
AdamOptimizerWithDecay(n_epochs, T; η₁, η₂, kwargs...)  # → (algorithm, linesearch)
```

puts the decay in a `SimpleSolvers.DecayingStatic` line search, to be splatted
into GO's `Optimizer(x, problem; method...)`. GML's `Optimizer`
(`src/optimizers/optimizer.jl:79`) carries a scalar `step_size` and computes the
schedule itself in `_current_step_size`; consuming GO's pairing means routing
GML's step size through a `LinesearchMethod` first. There are also ~12 call
sites under `docs/src/` and `scripts/`, and two documented behaviour changes for
migrated calls: the default element type goes `Float32` → `Float64` (GML took
`T` from `η₁`'s `Float32` literal), and `η₁`/`η₂`/`ρ₁`/`ρ₂`/`δ` become keywords
spelled `η₁`/`η₂`/`β₁`/`β₂`/`δ`.

[GO#34][go34] — which line search `AdamWithEuclideanDecay` defaults to — is still
open. It does not block anything here.

---

## 6. Order of execution

```
Phase 0  ── BLOCKING and entirely upstream:
            SimpleSolvers 0.12.1, GeometricIntegratorsBase 0.6.3,
            GeometricIntegrators 0.18.2  → unblocks resolution (R9)
            GeometricOptimizers 0.2.1    → unblocks the suite     (R8)

Phase 1 (target-form Project.toml)          ── done
Phase 2 (docs, retractions.md @ref)         ── done, unverifiable until Phase 0
§3      (1.10 import fix + optimizer move)  ── done
Phase 3 (1.10/1.13 confirmation)            ── needs Phase 0
Phase 5 (#234)                              ── separate PR
§5      (move traversal to GO)              ── follow-up; GO branch + GML issue
```

Every GML-side item is done. Nothing further can go green until the four
upstream releases land, and no amount of GML-side work substitutes for them.

---

## 7. Decisions (questions by claude, answers by benedict-96)

1. **Tag and register GO v0.2.0 now; if not, is dropping Julia 1.10 acceptable
   as an interim?** → Dropping 1.10 is acceptable, provided an issue records
   that it must be restored. Fix the other CI jobs first. *(Moot: GO 0.2.0 was
   registered on 2026-08-15 and 1.10 was never dropped.)*
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
   appropriate spot and tag michakraus. *(It does not: §2 removes bounds from
   type parameters, not from what the constructors accept. So: v0.2.1.)*
7. **Does "optimizer functionality belongs in GO" extend to
   `AdamOptimizerWithDecay` and the §5 traversal?** → Yes, all of it. Branch GO
   off `main`, move it there, delete from GML, and open a GML issue referencing
   the GO PR, integration left as a future task.
8. **If R8 needs an upstream Julia fix, may the test helpers be restructured
   temporarily?** → Yes, hand-in-hand with a new issue describing the problem.
   *(Moot: no Julia fix is needed and the tests were not touched.)*
9. **Per-testset progress markers in `runtests.jl`?** → Yes, but tag michakraus
   to inform him and ask whether he agrees.

---

## 8. Verification checklist

> [!IMPORTANT]
> The different Julia versions are available locally under `~/.julia/juliaup/`.
> Local checks use a scratch environment outside the repo that `dev`s the three
> local packages plus GML, so `git status` stays clean:
>
> ```julia
> # julia +1.12 --project=/tmp/gml-ci
> using Pkg
> Pkg.develop([PackageSpec(path = "/Users/mkraus/Julia/SimpleSolvers"),
>              PackageSpec(path = "/Users/mkraus/Julia/GeometricIntegratorsBase"),
>              PackageSpec(path = "/Users/mkraus/Datashare/Julia/GeometricIntegrators"),
>              PackageSpec(path = pwd())])
> ```

**R9 / resolution**
* [x] Target form applied: no `[sources]`, `julia = "1.10"`, 1.10 in the matrix,
      `GeometricIntegrators = "0.18.2"` bound added.
* [x] GO v0.2.0 visible in General; resolves to `GeometricOptimizers v0.2.0`
      from the registry, not from a path.
* [x] Resolves locally on 1.12.6 and 1.10.11 against the dev'd SS 0.12.1 /
      GIB 0.6.3 / GI 0.18.2.
* [ ] SimpleSolvers 0.12.1, GeometricIntegratorsBase 0.6.3 and
      GeometricIntegrators 0.18.2 registered; CI resolves without `dev`.
* [ ] All three 1.10 jobs pass.

**R8**
* [x] Cause located by profiling, not inferred from backtraces (§2).
* [x] Fix verified: repro goes from never completing to ~14.5 s cold / 6.5 ms
      warm, measured twice against a throwaway GO copy and once against the
      branch below.
* [x] GML-side traversal rewrite measured and rejected as ineffective.
* [x] Fix opened against GO on `inference-friendly-cache-parameters`, with GO's
      own suite green and a regression test that pins the parameters as
      unbounded.
* [ ] GO 0.2.1 released with the six struct bounds removed. No inner
      constructors: the outer constructors already enforce the invariant (§2).
* [ ] 1.12 × ubuntu completes in roughly the ~30 min it takes on `main` — a job
      that passes after several hours has outlasted R8, not fixed it.
* [ ] All three 1.12 jobs pass.

**Julia 1.10**
* [x] Blanket `using GeometricOptimizers` replaced by an explicit import list.
* [x] Optimizer machinery moved to `src/optimizers/optimizer.jl`, after the
      manifolds, so `_gml_rgrad` dispatches on GML's `Manifold`.
* [x] GML precompiles and loads on 1.10.11 and still on 1.12.6.
* [ ] Test suite run on 1.10.

**Docs**
* [x] Doctests excluded from `test/runtests.jl`; former examples covered under
      `test/`.
* [x] `docs/src/GeometricMachineLearning.bib` free of duplicate keys.
* [x] All `@ref` targets audited against all `@docs` entries; the seven
      unresolvable ones fixed (six de-referenced, `AdamOptimizerWithDecay`'s
      `@docs` entry restored) and the audit re-run clean.
* [ ] Documentation workflow green (every `@docs` entry and tutorial executes).
* [ ] PDF workflow green.
* [ ] `docs/Makefile`'s `test_docs` target passes — needs the local SS/GIB/GI
      `dev`'d into `docs/` too until Phase 0 lands.

**1.13**
* [x] 1.13 and nightly marked `experimental: true` in `CI.yml`; nightly stays so.
* [x] `GenericLinearAlgebra` absent from the resolved manifest (R3).
* [ ] Full matrix confirmed to resolve `GeometricIntegrators 0.18.2`.
* [ ] 1.13 green and de-experimentalised, or the linked upstream issue opened.

**Follow-ups**
* [ ] michakraus informed about the `runtests.jl` progress markers.
* [x] GO branch for the `AdamOptimizerWithDecay` move: [GO#33][go33] — merged,
      shipped in 0.2.0.
* [ ] [GO#34][go34] answered: `AdamWithEuclideanDecay`'s default line search
      confirmed.
* [ ] `AdamOptimizerWithDecay` deleted from `src/optimizers/optimizer.jl` and
      call sites moved to `Optimizer(x, problem; AdamOptimizerWithDecay(n)...)` —
      blocked on the §5 traversal move, see §5.
* [ ] GO branch opened for the §5 traversal (`_make_optimizer_cache`,
      `_make_optimizer_state`, `_tree_optim_step!`, `_leaf_optim_step!`,
      `GMLEuclideanState`); GML issue opened referencing both GO PRs.
* [ ] Issue opened for #234 Option C.
* [ ] Issue #234 closed by a PR that *deletes* the `b16267ea` bridge methods
      rather than adding more.
* [ ] `git status` clean; no dev artifacts in the diff.
