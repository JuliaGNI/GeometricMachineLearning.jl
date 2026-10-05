using Aqua
using ExplicitImports
using GeometricMachineLearning
using Test

# Aqua's package-level checks, all eight that `Aqua.test_all` enables by default.
#
# `Aqua.test_all` wraps each of its checks in a `@testset` of its own and adds no enclosing one, and
# a testset with no parent finalises as soon as it closes -- so called bare, the first failing check
# throws a `TestSetException` and the rest never run. On the suite path the `@safetestset` in
# `runtests.jl` is that parent. The `@testset` below is that parent when this file is run on its
# own.
#
# `ambiguities` and `piracies` were switched off until 0.9, with a count gate for each in their
# place: three pirated 3-tensor functors on `AbstractNeuralNetworks`' `Dense` and `Linear`, and one
# ambiguity between the first of them and `Affine`'s. `AbstractNeuralNetworks` 0.9 defines those
# methods itself, so they are deleted here and both checks run. No method of this package claims an
# argument position wholesale on `PoissonTensor`, which is an `AbstractMatrix{T}` and would
# otherwise meet every special-array method another package writes against `AbstractMatrix`:
#
#   `*(𝕁::PoissonTensor{T}, v::Strided…)` takes a `Strided…` right-hand side, which excludes each
#       special array type `X` that ArrayLayouts, FillArrays, Symbolics and GeometricOptimizers
#       define a `*(::AbstractMatrix, ::X)` for, and still covers everything the package builds.
#   `getindex(𝕁::PoissonTensor, i::Int, j::Int)` types its indices, which excludes the `Block`,
#       `BlockIndex` and `BandRangeType` methods BandedMatrices and BlockArrays add to
#       `AbstractMatrix`. Neither of those two loads with this package alone, so an untyped index
#       pair here is a collision visible only from inside the suite.
@testset "Aqua" begin
    Aqua.test_all(GeometricMachineLearning)
end

# `ExplicitImports` answers a question Aqua does not ask, and the reason to gate on it is not
# tidiness: a `using` or an `import` that nothing needs keeps a dependency reachable from the module
# file, so Aqua's `stale_deps` above sees a package in use that nothing uses. That check only means
# something once these two are clean.
#
# Three of the seven checks run: `no_stale_explicit_imports`, `all_explicit_imports_via_owners` and
# `no_self_qualified_accesses`. The first two were red before this file gained them -- 13 stale
# explicit imports, and `StateVariable` imported from `GeometricSolutions`, which re-exports it,
# rather than from `GeometricBase`, which defines it. The third already passed.
#
# THE OTHER FOUR, AND WHY EACH IS OFF.
#
#   no_implicit_imports              37 names arrive through a bare `using`, across thirteen packages
#                                    -- `norm`, `@kernel`, `rrule`, `pullback` and the rest. Naming
#                                    every one is a change to the whole module header and a judgement
#                                    per name, not a by-product of a dead-code pass.
#   all_explicit_imports_are_public  9 names this package imports are not marked `public` upstream
#                                    -- `Architecture`, `AbstractExplicitLayer`, `_compute_loss`,
#                                    `description`, `dim` among them. Each is deliberate and most
#                                    are re-exported here; the fix is upstream declaring them, not
#                                    this package importing less.
#   all_qualified_accesses_via_owners  2: `GeometricOptimizers.Gradient` and
#                                    `GeometricOptimizers.direction`, both owned by `SimpleSolvers`.
#                                    Reaching them through `GeometricOptimizers` is how the rest of
#                                    `src/optimizers/optimizer.jl` is written.
#   all_qualified_accesses_are_public  24, the same class as the 9 above: `KernelAbstractions.zeros`,
#                                    `ForwardDiff.jacobian`, `GeometricOptimizers.momentum` and so on.
#
# Each of the four is a real backlog item rather than a taste, and none of them is this branch's.
#
# WHAT THIS GATE DOES NOT CATCH, and one way it can go red on its own.
#
# It reads imports, never definitions. A dead *definition* re-added under `src/` leaves all three
# checks green, so this closes only the import half of the pass that added it.
#
# `all_explicit_imports_via_owners` also has the property the `ambiguities` comment above rejects:
# it asks *which* upstream module defines a name, so it goes red when a name moves between
# `GeometricBase`, `GeometricOptimizers` and `AbstractNeuralNetworks` with nothing in `src/` having
# changed -- which is exactly how `StateVariable` got here. `ExplicitImports = "1.15"` admits any
# 1.x, so a minor release that sharpens stale detection does the same. Unlike the ambiguity count,
# a red here names the import and the module to take it from, so it stays a gate: the fix is one
# line and the message says which. `test_explicit_imports` opens its own `@testset`, so this call
# needs no wrapper.
test_explicit_imports(GeometricMachineLearning;
    no_implicit_imports = false,
    all_explicit_imports_are_public = false,
    all_qualified_accesses_via_owners = false,
    all_qualified_accesses_are_public = false)
