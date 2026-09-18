# `parameterlength(d::MultiHeadAttention{M, M, true})` counts parameters with `÷` alone, which is
# exact by construction for every size that fits in an `Int`. Exactness at every size matters
# because `MultiHeadAttention` stores only `n_heads::Int` and `activation`, so `parameterlength`
# can be evaluated at a scale no real layer could allocate -- a scale where `Float64` no longer
# represents the intermediate product exactly. Same guard as `PSDLayer`'s, in
# `test/layers/psd_parameterlength.jl`. See `CHANGELOG.md` for the fix this guards.

using GeometricMachineLearning
using Test

GML = GeometricMachineLearning

@testset "parameterlength(::MultiHeadAttention{M,M,true}) is exact at ordinary sizes" begin
    for (M, n) in ((4, 2), (12, 3), (100, 5))
        d = GML.MultiHeadAttention{M, M, true, true, GML.VectorSoftmax}(n, GML.VectorSoftmax())
        expected = 3 * M^2 - (3 * M * (M + n)) ÷ (2 * n)
        @test GML.parameterlength(d) == expected
    end
end

@testset "parameterlength(::MultiHeadAttention{M,M,true}) is exact beyond the Float64 mantissa" begin
    # `3*M*(M + n)` is about `2.1e16` here, past the `2^53` up to which a `Float64` holds every
    # integer, so the same count taken through a `Float64` division lands on 19296855159637520.
    # The exact value, checked against `BigInt` arithmetic, is 19296855159637518.
    M = 83_767_764
    n = 6
    d = GML.MultiHeadAttention{M, M, true, true, GML.VectorSoftmax}(n, GML.VectorSoftmax())
    @test GML.parameterlength(d) == 19_296_855_159_637_518
end
