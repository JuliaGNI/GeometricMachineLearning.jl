# `parameterlength(::PSDLayer{M, N})` computes an exact combinatorial parameter count with `÷`
# alone, which is exact by construction for any size that itself fits in `Int`. Exactness at
# every size matters because `PSDLayer{M, N}` is a singleton (it stores nothing sized by `M` or
# `N`), so `parameterlength` can be called at a scale no real layer could ever allocate -- a scale
# where `Float64` no longer represents the intermediate product exactly. See `CHANGELOG.md` for the
# fix this guards.

using GeometricMachineLearning
using Test

GML = GeometricMachineLearning

@testset "parameterlength(::PSDLayer) is exact at ordinary sizes" begin
    for (M, N) in ((4, 6), (6, 4), (20, 30), (30, 20))
        l = GML.PSDLayer{M, N}()
        M2, N2 = M ÷ 2, N ÷ 2
        expected = N > M ? M2 * N2 - (M2 * (M2 + 1)) ÷ 2 : N2 * M2 - (N2 * (N2 + 1)) ÷ 2
        @test GML.parameterlength(l) == expected
    end
end

@testset "parameterlength(::PSDLayer) is exact beyond the Float64 mantissa" begin
    # The count is about `1.1e16` here, past the `2^53` up to which a `Float64` holds every
    # integer, so the same count taken through a `Float64` division lands on 11427319538133784.
    # The exact value, checked against `BigInt` arithmetic, is 11427319538133785.
    M2 = 151_177_507
    M = 2 * M2
    N = M + 4
    l = GML.PSDLayer{M, N}()
    @test GML.parameterlength(l) == 11_427_319_538_133_785
end
