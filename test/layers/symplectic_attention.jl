using GeometricMachineLearning
using GeometricMachineLearning: params
using KernelAbstractions
using LinearAlgebra: I
using Test
using Zygote: jacobian
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

# The Jacobian of the layer on a sequence `(Q, P)` of `L` vectors preserves the canonical form
# that pairs `vec(Q)` with `vec(P)`, for either activation.
function sequence_jacobian(l, ps, Q::AbstractMatrix, P::AbstractMatrix)
    n = length(Q)
    function f(w)
        out = l((q = reshape(w[1:n], size(Q)), p = reshape(w[(n + 1):end], size(P))), ps)
        vcat(vec(out.q), vec(out.p))
    end
    jacobian(f, vcat(vec(Q), vec(P)))[1]
end

@testset "symplecticity ($(nameof(typeof(activation))), symmetric = $symmetric)" for activation in (
        MatrixSoftmax(), VectorSoftmax()),
    symmetric in (true, false)

    d, L = 2, 4
    Ω = [zeros(d * L, d * L) I; -I zeros(d * L, d * L)]
    for l in (SymplecticAttentionQ(2d; symmetric = symmetric, activation = activation),
        SymplecticAttentionP(2d; symmetric = symmetric, activation = activation))
        ps = params(NeuralNetwork(Chain(l), CPU(), Float64)).L1
        J = sequence_jacobian(l, ps, randn(d, L), randn(d, L))
        @test J' * Ω * J ≈ Ω atol = 1e-12
        @test !(J ≈ I)
    end
end
