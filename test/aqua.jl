using Aqua
using ExplicitImports
using GeometricMachineLearning
using Test

# Aqua's package-level checks. Six of the eight `Aqua.test_all` enables by default run; `ambiguities`
# and `piracies` are switched off by name, and the reason is below rather than in the changelog
# alone, because switching one back on turns the suite red on the spot.
#
# `Aqua.test_all` wraps each of its checks in a `@testset` of its own and adds no enclosing one, and
# a testset with no parent finalises as soon as it closes -- so called bare, the first failing check
# throws a `TestSetException` and the rest never run. On the suite path the `@safetestset` in
# `runtests.jl` is that parent. The `@testset` below is that parent when this file is run on its
# own, and it is also what lets the two counts at the end be further assertions rather than
# top-level `@test`s that abort the file the moment one fails.
#
# WHAT IS SWITCHED OFF, AND WHY IT IS NOT A `broken = true`.
#
# `piracies` reports 3 methods, and all 3 are genuine under Aqua's definition -- the function and
# every argument type belong to other modules. They are not a count waiting to be triaged. All
# three are the same thing: a three-argument functor for a *layer* type that this package does not
# own, `AbstractNeuralNetworks`' `Dense` and `Linear`, added so that the layer accepts a 3-tensor
# input as well as a matrix.
#
#     Dense{M, N, true}   on an AbstractArray{T, 3}    src/layers/resnet.jl:63
#     Dense{M, N, false}  on an AbstractArray{T, 3}    src/layers/resnet.jl:67
#     Linear{M, N}        on an AbstractArray{T, 3}    src/layers/resnet.jl:71
#
# Closing these means either upstream gaining the 3-tensor methods or this package wrapping the two
# layer types, and both are an API change rather than a tidy-up. They are *B7* under
# `## Open Issues` in `CHANGELOG.md` with that reasoning.
#
# `ambiguities` reports 1, and the same 1 whether this package is loaded alone or with everything
# `runtests.jl` loads before this file. It is a `Dense{M, N, true}` functor against
# `AbstractNeuralNetworks.Affine`'s, and it is benign: `Dense` is not a subtype of `Affine` and the
# two have no common instance, so no call can reach the pair. It is the same `resnet.jl:63` method
# as the first piracy above.
#
# The two settings agreeing is what makes the gate below trustworthy, and it holds because no
# method of this package claims an argument position wholesale on `PoissonTensor`. That type is an
# `AbstractMatrix{T}`, so such a method meets every special-array method another package writes
# against `AbstractMatrix`:
#
#   `*(𝕁::PoissonTensor{T}, v::Strided…)` takes a `Strided…` right-hand side, which excludes each
#       special array type `X` that ArrayLayouts, FillArrays, Symbolics and GeometricOptimizers
#       define a `*(::AbstractMatrix, ::X)` for, and still covers everything the package builds.
#   `getindex(𝕁::PoissonTensor, i::Int, j::Int)` types its indices, which excludes the `Block`,
#       `BlockIndex` and `BandRangeType` methods BandedMatrices and BlockArrays add to
#       `AbstractMatrix`. Neither of those two loads with this package alone, so an untyped index
#       pair here is a collision visible only from inside the suite.
#
# `ambiguities` still stays off, because 1 is not 0 and Aqua's check has no way to exempt a pair.
# The assertion below replaces it and is strictly better here: it names the number, so it fails
# both when an ambiguity is added and when this one is closed without the comment going with it.
#
# Marking either check `broken = true` would leave a check that reports success while the defect
# stands, which is the failure mode this test suite's guards exist to remove.
#
# `unbound_args` is the one check here whose verdict depends on the Julia version: it passes on
# `min`, `1` and `pre`, and fails on nightly over one method. That is *B9* under `## Open Issues`.
@testset "Aqua" begin
    Aqua.test_all(GeometricMachineLearning; ambiguities = false, piracies = false)

    # A switched-off check detects nothing, so the count it would have reported drifts unobserved,
    # and so does every `src` line named above. This is the gate. It fails when a piracy is added,
    # and it fails when one is removed without the entry above and in `CHANGELOG.md` going with it.
    @test length(Aqua.Piracy.hunt(GeometricMachineLearning)) == 3

    # The ambiguity count gets the same gate, and it can carry one because the single pair is
    # between a method of this package and one of `AbstractNeuralNetworks`'. The count is the same
    # in isolation and in the suite, and nothing outside this repository's own `[compat]` moves it
    # -- so a red here is a regression, not an unrelated upstream upgrade and not a reordering of
    # this suite. A count that moved with ArrayLayouts', FillArrays', Symbolics',
    # GeometricOptimizers', BandedMatrices' and BlockArrays' versions, and with *which* of them a
    # given process had loaded, could not tell those apart, and could not be asserted.
    @test length(Test.detect_ambiguities(GeometricMachineLearning; recursive = true)) == 1
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
#                                    `description`, `dim` among them. One fewer than before, because
#                                    `assign_columns` left with the `GeometricOptimizers` 0.8 bump
#                                    and `orthonormal_columns`, which replaces it, is public there.
#                                    Each is deliberate and most are re-exported here; the fix is
#                                    upstream declaring them, not this package importing less.
#   all_qualified_accesses_via_owners  2: `GeometricOptimizers.Gradient` and
#                                    `GeometricOptimizers.direction`, both owned by `SimpleSolvers`.
#                                    Reaching them through `GeometricOptimizers` is how the rest of
#                                    `src/optimizers/optimizer.jl` is written.
#   all_qualified_accesses_are_public  24, the same class as the 10 above: `KernelAbstractions.zeros`,
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
