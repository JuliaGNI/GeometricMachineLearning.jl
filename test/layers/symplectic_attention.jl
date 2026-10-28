using GeometricMachineLearning
using GeometricMachineLearning: params
using KernelAbstractions
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

# A single sample gives the column of the one-column matrix result, as a vector or a pair of vectors.
@testset "single sample ($T, symmetric = $symmetric)" for T in REAL_ELTYPES,
    symmetric in (true, false)

    for l in (SymplecticAttentionQ(6; symmetric = symmetric),
        SymplecticAttentionP(6; symmetric = symmetric))
        ps = params(NeuralNetwork(Chain(l), CPU(), T)).L1
        z = randn(T, 6)
        zm = reshape(z, :, 1)
        ym = l(zm, ps)

        y = @inferred l(z, ps)
        @test y isa Vector{T}
        @test y ≈ vec(ym) rtol = 8 * eps(T)

        nt = @inferred l((q = z[1:3], p = z[4:6]), ps)
        @test nt isa NamedTuple{(:q, :p), Tuple{Vector{T}, Vector{T}}}
        @test vcat(nt.q, nt.p) ≈ vec(ym) rtol = 8 * eps(T)
        @test !(y ≈ z)
    end
end
