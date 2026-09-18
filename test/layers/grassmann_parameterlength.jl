# `parameterlength(::GrassmannLayer{M, N})` returned a `UnitRange` whenever `M >= N`, because a
# colon stood where the product belongs: `(M - N):N` instead of `(M - N) * N`. Nothing asserted the
# count, so it went unnoticed -- `parameterlength` is used for reporting rather than for sizing an
# array, and a range compares equal to nothing that was being compared.

using GeometricMachineLearning
using Test

GML = GeometricMachineLearning

@testset "parameterlength(::GrassmannLayer) is an integer count" begin
    # Derived independently of `src`: the Grassmann manifold Gr(n, k) of k-dimensional subspaces of
    # an n-dimensional space has dimension k * (n - k), which is the number of entries of the
    # (n - k) x k matrix that gives a subspace's coordinates in the complement of a reference one.
    grassmann_dimension(n, k) = k * (n - k)

    for (M, N) in ((4, 6), (6, 4), (10, 4), (4, 10), (5, 5))
        l = GrassmannLayer(M, N)
        n, k = max(M, N), min(M, N)

        @test GML.parameterlength(l) isa Integer
        @test GML.parameterlength(l) == grassmann_dimension(n, k)
    end
end

@testset "the count is symmetric in its two sizes" begin
    # Gr(n, k) and the layer built the other way round describe subspaces of the same pair of
    # dimensions, so the two counts agree. The broken branch made them disagree: the `M >= N` case
    # returned a range and the `N > M` case an integer.
    for (M, N) in ((4, 6), (10, 4), (3, 7))
        @test GML.parameterlength(GrassmannLayer(M, N)) ==
              GML.parameterlength(GrassmannLayer(N, M))
    end
end
