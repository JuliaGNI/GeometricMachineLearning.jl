using Test
using GeometricMachineLearning
using GeometricMachineLearning: _custom_mul, params
using LinearAlgebra: norm
using Zygote: gradient
import Random

Random.seed!(1234)

@doc raw"""
See `test/attention_layer/symplectic_attention_network_parameters_gradient.jl` for why a gradient
taken with respect to a `NetworkParameters` wrapper can lose its `SymmetricMatrix` leaf while the
same gradient taken with respect to the bare wrapped `NamedTuple` (`params(ps)`) never does, and why
the two still agree numerically regardless. `single_multiplication` uses its `SymmetricMatrix`
argument once; `double_multiplication` uses it twice.
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

function single_multiplication(a::AT, _ps::Union{
        NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    _custom_mul(a, _ps.L1.A)
end

function double_multiplication(a::AT, _ps::Union{
        NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    _custom_mul(_ps.L1.A, _custom_mul(a, _ps.L1.A))
end

S = rand(SymmetricMatrix, 4)
ps = NetworkParameters((L1 = (A = S,),))
t = rand(4, 4)

@testset "_custom_mul: NetworkParameters gradient structure" begin
    @testset "single_multiplication (A used once: structure kept)" begin
        network_parameters_gradient_structure_test(
            _ps -> single_multiplication(t, _ps), ps, true)
    end
    @testset "double_multiplication (A used twice: structure lost)" begin
        network_parameters_gradient_structure_test(
            _ps -> double_multiplication(t, _ps), ps, false)
    end
end
