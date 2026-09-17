using Aqua
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
# own, and it is also what lets the piracy count at the end be a second assertion rather than a
# top-level `@test` that aborts the file the moment it fails.
#
# WHAT IS SWITCHED OFF, AND WHY IT IS NOT A `broken = true`.
#
# `piracies` reports 12 methods, and all 12 are genuine under Aqua's definition -- the function and
# every argument type belong to other modules. They are not a count waiting to be triaged: each has
# a witness, a call whose behaviour changes when this package is loaded. Three of them are on
# `Base`, so they change every Julia process that loads this one:
#
#     1.0 + (2.0,)      MethodError without GML, 3.0 with it              src/utils.jl:61
#     [1.0] + (2.0,)    MethodError without GML, 3.0 with it -- a scalar, src/utils.jl:67
#                       silently discarding every element but the first
#     (q, p) ≈ (q, p)   MethodError without GML, true with it             src/utils.jl:163
#
# `ambiguities` reports 23, of which 17 are `PoissonTensor * v` against left-multiply methods in
# ArrayLayouts, FillArrays, Symbolics and GeometricOptimizers.
#
# Fixing either set is a change to `src/` that this file does not own, and the two lists are
# recorded under `## Open Issues` in `CHANGELOG.md` with the witnesses. Marking them `broken = true`
# would leave a check that reports success while the defects stand, which is the failure mode this
# test suite's guards exist to remove. Six checks that fail on a real regression are worth more than
# eight that are all switched off.
#
# `unbound_args` is the one check here whose verdict depends on the Julia version: it passes on
# `min`, `1` and `pre`, and fails on nightly over one method. That is *B9* under `## Open Issues`.
@testset "Aqua" begin
    Aqua.test_all(GeometricMachineLearning; ambiguities = false, piracies = false)

    # A switched-off check detects nothing, so the count it would have reported drifts unobserved,
    # and so does every `src` line named above. This is the gate. It fails when a piracy is added,
    # and it fails when one is removed without the entry above and in `CHANGELOG.md` going with it.
    #
    # `ambiguities` gets no such gate. Seventeen of its 23 are against methods in ArrayLayouts,
    # FillArrays, Symbolics and GeometricOptimizers, so the count moves with those packages'
    # versions rather than with anything in this tree, and an exact assertion would turn the suite
    # red on an unrelated upgrade.
    @test length(Aqua.Piracy.hunt(GeometricMachineLearning)) == 12
end
