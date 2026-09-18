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
# `ambiguities` reported 23. Four were the loss functors against `AbstractNeuralNetworks`'
# `(::NetworkLoss)(::NeuralNetwork, …)`, and one was `_GMLGradient` against `SimpleSolvers`'
# `(::Gradient{T})(::AbstractVector{T})` -- all five fixed by typing the losses' first parameter and
# adding the missing `_GMLGradient` method (`optimizer.jl`). The remaining 18 are `PoissonTensor *
# v` against left-multiply methods in ArrayLayouts, FillArrays, Symbolics and GeometricOptimizers
# (17 of them -- out of scope here, see `## Open Issues`), plus one benign `Dense`/`Affine` pair
# with no witness. `ambiguities` stays off: 18 is not 0, and an exact assertion on the 17 would turn
# the suite red on an unrelated upstream upgrade the way the piracy count does not.
#
# Fixing the piracy set is a change to `src/` that this file does not own, and it is recorded under
# `## Open Issues` in `CHANGELOG.md` with the witnesses. Marking it `broken = true` would leave a
# check that reports success while the defect stands, which is the failure mode this test suite's
# guards exist to remove. Six checks that fail on a real regression are worth more than eight that
# are all switched off.
#
# `unbound_args` is the one check here whose verdict depends on the Julia version: it passes on
# `min`, `1` and `pre`, and fails on nightly over one method. That is *B9* under `## Open Issues`.
@testset "Aqua" begin
    Aqua.test_all(GeometricMachineLearning; ambiguities = false, piracies = false)

    # A switched-off check detects nothing, so the count it would have reported drifts unobserved,
    # and so does every `src` line named above. This is the gate. It fails when a piracy is added,
    # and it fails when one is removed without the entry above and in `CHANGELOG.md` going with it.
    #
    @test length(Aqua.Piracy.hunt(GeometricMachineLearning)) == 12

    # `ambiguities` itself gets no exact gate on the 17 `PoissonTensor` ones -- they move with
    # ArrayLayouts', FillArrays', Symbolics' and GeometricOptimizers' own versions, not with
    # anything in this tree. But the total is still asserted, so a new ambiguity introduced here
    # (rather than upstream) does not silently join that pile: it currently accounts for exactly
    # 18, all of them either the `PoissonTensor` set or the one benign `Dense`/`Affine` pair.
    @test length(Test.detect_ambiguities(GeometricMachineLearning; recursive = true)) == 18
end
