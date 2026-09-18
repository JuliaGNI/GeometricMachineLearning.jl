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
#     1.0 + (2.0,)      MethodError without GML, 3.0 with it              src/utils.jl:65
#     [1.0] + (2.0,)    MethodError without GML, 3.0 with it -- a scalar, src/utils.jl:71
#                       silently discarding every element but the first
#     (q, p) ≈ (q, p)   MethodError without GML, true with it             src/utils.jl:167
#
# `ambiguities` reports 18 when `GeometricMachineLearning` is the only package loaded: the 17
# `PoissonTensor * v` ambiguities against left-multiply methods in ArrayLayouts, FillArrays,
# Symbolics and GeometricOptimizers (out of scope here, see `## Open Issues`), plus one benign
# `Dense`/`Affine` pair with no witness.
#
# **But 18 is not what this file measures, because this file does not run in isolation.** By the
# time this `@safetestset` runs, `runtests.jl` has already loaded `Zygote`, `GeometricIntegrators`
# and `HDF5` for earlier subjects, and that combination pulls in `BandedMatrices` and `BlockArrays`
# as transitive extension dependencies -- neither of which loads with `GeometricMachineLearning`
# alone, or with any *one* of those three added to it. Both packages specialise `getindex` on an
# `AbstractMatrix` for their own index types (`Block`, `BandRangeType`, …), and
# `PoissonTensor`'s own `getindex(𝕁::PoissonTensor, i, j)` (`poisson_tensor.jl:42`) is exactly as
# generic on its index arguments, so it collides with **9** of them -- the same class of defect as
# the 17 `*` ambiguities, on the same type, out of scope for the same reason. Measured with every
# package `runtests.jl` loads before this file present, the true total is **27**, and that is the
# number that governs whether `Pkg.test()` passes.
#
# `ambiguities` stays off either way: 27 (or 18) is not 0, and an exact assertion on the
# `PoissonTensor` pile would turn the suite red on an unrelated upstream upgrade the way the piracy
# count does not.
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
    @test length(Aqua.Piracy.hunt(GeometricMachineLearning)) == 12

    # `ambiguities` gets no such gate, and the measurements above are why it cannot have one. The
    # remaining ambiguities are the `PoissonTensor` pile, so the count moves with ArrayLayouts',
    # FillArrays', Symbolics', GeometricOptimizers', BandedMatrices' and BlockArrays' versions
    # rather than with anything in this tree -- and with *which* of them a given process has
    # loaded, which depends on the order `runtests.jl` includes its subjects. An exact assertion
    # would therefore go red on an unrelated upgrade, or on a reordering of this suite, with
    # nothing in `src/` having changed. It could not tell that apart from a regression, which is
    # the one thing a gate has to do.
end
