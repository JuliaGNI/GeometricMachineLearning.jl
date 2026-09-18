# `parameterlength(::PSDLayer{M, N})` used to route an integer count through `Float64` division
# and back through `Int(...)`. That is exact for every size the rest of the suite constructs, but
# not for every size the *type* accepts: `PSDLayer{M, N}` is a singleton (it stores nothing sized by
# `M` or `N`), so `parameterlength` can be called at a scale no real layer could ever allocate, and
# there the `Float64` intermediate silently rounds -- `Int(...)` on a value like `...784.9999999`
# gives the wrong integer instead of throwing. The rewrite below computes the same combinatorial
# count with `÷` alone, which is exact by construction for any size that itself fits in `Int`.

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

@testset "parameterlength(::PSDLayer) is exact where the old Float64 path was not" begin
    # M2 = 151_177_507: `Int(M2 * (N2 - (M2 + 1) / 2))` rounds to 11427319538133784 here; the exact
    # value (checked against `BigInt` arithmetic) is 11427319538133785.
    M2 = 151_177_507
    M = 2 * M2
    N = M + 4
    l = GML.PSDLayer{M, N}()
    @test GML.parameterlength(l) == 11_427_319_538_133_785
end
