# Same defect as `PSDLayer` (see `test/layers/psd_parameterlength.jl`): `parameterlength(d::
# MultiHeadAttention{M, M, true})` routed an integer count through `Float64` division and back
# through `Int(...)`, exact for every size the rest of the suite constructs but not for every size
# the type accepts. `MultiHeadAttention` stores only `n_heads::Int` and `activation`, so
# `parameterlength` can be evaluated at a scale no real layer could allocate, and there the old
# `Float64` path silently rounds to the wrong integer.

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

@testset "parameterlength(::MultiHeadAttention{M,M,true}) is exact where the old Float64 path was not" begin
    # M = 83_767_764, n_heads = 6: `Int(3*M^2 - 3*M*(M + n)/(2*n))` rounds to 19296855159637520
    # here; the exact value (checked against `BigInt` arithmetic) is 19296855159637518.
    M = 83_767_764
    n = 6
    d = GML.MultiHeadAttention{M, M, true, true, GML.VectorSoftmax}(n, GML.VectorSoftmax())
    @test GML.parameterlength(d) == 19_296_855_159_637_518
end
