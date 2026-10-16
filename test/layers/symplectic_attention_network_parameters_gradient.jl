using Test
using GeometricMachineLearning
using GeometricMachineLearning: _mul, _custom_transpose, params
import Random

include("../helpers/network_parameters_gradient_structure.jl")

Random.seed!(1234)

function symplectic_attention(z::NamedTuple{(:q, :p), Tuple{AT, AT}},
        _ps::Union{NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    expPAP = exp.(_mul(_mul(_custom_transpose(z.p), _ps.L1.A), z.p))
    (q = z.q + _mul(_mul(_ps.L1.A, z.p), 2 * expPAP) / sum(expPAP), p = z.p)
end

function symplectic_attention_simplified(z::NamedTuple{(:q, :p), Tuple{AT, AT}},
        _ps::Union{NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    (q = z.p + _mul(_mul(z.p, _ps.L1.A), z.p), p = z.p)
end

function symplectic_linear_map(z::NamedTuple{(:q, :p), Tuple{AT, AT}},
        _ps::Union{NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    (q = z.q + _mul(_ps.L1.A, z.p), p = z.p)
end

S = rand(SymmetricMatrix, 2)
ps = NetworkParameters((L1 = (A = S,),))
t = (q = rand(2, 2), p = rand(2, 2))

@testset "Symplectic attention: NetworkParameters gradient structure" begin
    @testset "symplectic_attention (two wrapper accesses)" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_attention(t, _ps), ps)
    end
    @testset "symplectic_attention_simplified (one wrapper access)" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_attention_simplified(t, _ps), ps)
    end
    @testset "symplectic_linear_map (one wrapper access)" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_linear_map(t, _ps), ps)
    end
end

# `Chain` reaches the layer through `values(_ps)` rather than through a `getproperty` on the
# wrapper, and the layer's body reads `A` twice.
@testset "SymplecticAttentionQ layer: NetworkParameters gradient structure" begin
    l = SymplecticAttentionQ(4; symmetric = true)
    nn = NeuralNetwork(Chain(l))
    network_parameters_gradient_structure_test(_ps -> nn(t, _ps), params(nn))
end
