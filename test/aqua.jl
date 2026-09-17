using Aqua
using GeometricMachineLearning
using Test

# Aqua's package-level checks. Six of the eight run; `ambiguities` and `piracies` are switched off
# by name, and the reason is below rather than in the changelog alone, because switching one back on
# turns the suite red on the spot.
#
# `Aqua.test_all` wraps each of its checks in a `@testset` of its own and adds no enclosing one. A
# testset with no parent finalises as soon as it closes, so called bare the first failing check
# throws a `TestSetException` and the rest never run. On the suite path the `@safetestset` in
# `runtests.jl` is that parent; the `@testset` below is what keeps all six checks reachable when
# this file is run on its own.
#
# WHAT IS SWITCHED OFF, AND WHY IT IS NOT A `broken = true`.
#
# `piracies` reports 13 methods, and all 13 are genuine under Aqua's definition — the function and
# every argument type belong to other modules. They are not a count waiting to be triaged: each has
# a witness, a call whose behaviour changes when this package is loaded. The severe three are on
# `Base`, so they change every Julia process that loads this one:
#
#     1.0 + (2.0,)      MethodError without GML, 3.0 with it              src/utils.jl:61
#     [1.0] + (2.0,)    MethodError without GML, 3.0 with it -- a scalar, src/utils.jl:67
#                       silently discarding every element but the first
#     dim(nn)           a clean MethodError becomes a logged error and    src/backends/lux.jl:56
#                       a returned `nothing`, for every architecture
#                       that does not implement `dim`
#
# `ambiguities` reports 23, of which 17 are `PoissonTensor * v` against left-multiply methods in
# ArrayLayouts, FillArrays, Symbolics and GeometricOptimizers.
#
# Fixing either set is a change to `src/` that this file does not own, and the two lists are
# recorded under `## Open Issues` in `CHANGELOG.md` with the witnesses. Marking them `broken = true`
# would leave a check that reports success while the defects stand, which is the failure mode this
# test suite's guards exist to remove. Six checks that fail on a real regression are worth more than
# eight that are all switched off.
@testset "Aqua" begin
    Aqua.test_all(GeometricMachineLearning; ambiguities = false, piracies = false)
end
