using Test
using GeometricMachineLearning
using GeometricMachineLearning: _custom_mul, _custom_transpose, params
import Random

include("../helpers/network_parameters_gradient_structure.jl")

Random.seed!(1234)

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
    @testset "symplectic_attention" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_attention(t, _ps), ps)
    end
    @testset "symplectic_attention_simplified" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_attention_simplified(t, _ps), ps)
    end
    @testset "symplectic_linear_map" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_linear_map(t, _ps), ps)
    end
end

# The same through a layer, which `Chain` reaches through `values(_ps)` rather than through a
# `getproperty` on the wrapper.
@testset "SymplecticAttentionQ layer: NetworkParameters gradient structure" begin
    l = SymplecticAttentionQ(4; symmetric = true)
    nn = NeuralNetwork(Chain(l))
    network_parameters_gradient_structure_test(_ps -> nn(t, _ps), params(nn))
end
