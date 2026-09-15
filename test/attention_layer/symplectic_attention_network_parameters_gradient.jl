using Test
using GeometricMachineLearning
using GeometricMachineLearning: _custom_mul, _custom_transpose, params
import Random

include("../network_parameters_gradient_structure.jl")

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
    @testset "symplectic_attention (two wrapper accesses: structure lost)" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_attention(t, _ps), ps, false)
    end
    @testset "symplectic_attention_simplified (one wrapper access: structure kept)" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_attention_simplified(t, _ps), ps, true)
    end
    @testset "symplectic_linear_map (one wrapper access: structure kept)" begin
        network_parameters_gradient_structure_test(
            _ps -> symplectic_linear_map(t, _ps), ps, true)
    end
end

# The layer keeps the structure although its body reads `A` twice. `Chain` performs the single
# `getproperty` on the wrapper (`_ps.L1`) and hands the layer a plain `NamedTuple`, and accesses
# below the wrapper do not count. The local binding at `src/layers/symplectic_attention.jl` is
# therefore not what saves this case: the same body written without it, reached through the same
# one wrapper access, also returns a `SymmetricMatrix`.
@testset "SymplecticAttentionQ layer: NetworkParameters gradient structure" begin
    l = SymplecticAttentionQ(4; symmetric = true)
    nn = NeuralNetwork(Chain(l))
    network_parameters_gradient_structure_test(_ps -> nn(t, _ps), params(nn), true)
end
