using Test
using GeometricMachineLearning
using GeometricMachineLearning: _custom_mul, _custom_transpose, params
using LinearAlgebra: norm
using Zygote: gradient
import Random

Random.seed!(1234)

@doc raw"""
A gradient taken with respect to a `NetworkParameters` wrapper comes back as a `NetworkParameters`
whose `SymmetricMatrix` leaf survives only when that leaf is used once in the loss expression, and
degrades to a plain `Matrix` when it is used more than once -- the wrapper's own gradient
accumulation across the repeated use, not the operation doing the using; see `CHANGELOG.md` for the
experiment. A gradient taken with respect to the bare wrapped `NamedTuple` (`params(ps)`) keeps the
structure either way.

This asserts, per function, which shape survives, and that the two gradients agree numerically
regardless of which one loses the structure.
"""
function network_parameters_gradient_structure_test(
        f, ps::NetworkParameters, preserves_structure::Bool)
    g_wrapped = gradient(_ps -> norm(f(_ps)), ps)[1]
    g_bare = gradient(_ps -> norm(f(_ps)), params(ps))[1]

    a_wrapped = g_wrapped.L1.A
    a_bare = g_bare.L1.A

    @test typeof(a_bare) <: SymmetricMatrix
    if preserves_structure
        @test typeof(a_wrapped) <: SymmetricMatrix
    else
        @test_broken typeof(a_wrapped) <: SymmetricMatrix
    end

    @test isapprox(Matrix(a_wrapped), Matrix(a_bare))
end

function symplectic_attention(z::NamedTuple{(:q, :p), Tuple{AT, AT}},
        _ps::Union{NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    expPAP = exp.(_custom_mul(_custom_mul(_custom_transpose(z.p), _ps.L1.A), z.p))
    (q = z.q + _custom_mul(_custom_mul(_ps.L1.A, z.p), 2 * expPAP) / sum(expPAP), p = z.p)
end

function symplectic_attention_simplified(z::NamedTuple{(:q, :p), Tuple{AT, AT}},
        _ps::Union{NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    (q = z.p + _custom_mul(_custom_mul(z.p, _ps.L1.A), z.p), p = z.p)
end

function symplectic_linear_map(z::NamedTuple{(:q, :p), Tuple{AT, AT}},
        _ps::Union{NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    (q = z.q + _custom_mul(_ps.L1.A, z.p), p = z.p)
end

S = rand(SymmetricMatrix, 2)
ps = NetworkParameters((L1 = (A = S,),))
t = (q = rand(2, 2), p = rand(2, 2))

@testset "Symplectic attention: NetworkParameters gradient structure" begin
    @testset "symplectic_attention (A used twice: structure lost)" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_attention(t, _ps), ps, false)
    end
    @testset "symplectic_attention_simplified (A used once: structure kept)" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_attention_simplified(t, _ps), ps, true)
    end
    @testset "symplectic_linear_map (A used once: structure kept)" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_linear_map(t, _ps), ps, true)
    end
end

@testset "SymplecticAttentionQ layer: NetworkParameters gradient structure" begin
    l = SymplecticAttentionQ(4; symmetric = true)
    nn = NeuralNetwork(Chain(l))
    network_parameters_gradient_structure_test(_ps -> nn(t, _ps), nn.params, true)
end
