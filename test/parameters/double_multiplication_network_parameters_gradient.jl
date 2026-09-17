using Test
using GeometricMachineLearning
using GeometricMachineLearning: _custom_mul
import Random

include("network_parameters_gradient_structure.jl")

Random.seed!(1234)

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
    @testset "single_multiplication (one wrapper access: structure kept)" begin
        network_parameters_gradient_structure_test(
            _ps -> single_multiplication(t, _ps), ps, true)
    end
    @testset "double_multiplication (two wrapper accesses: structure lost)" begin
        network_parameters_gradient_structure_test(
            _ps -> double_multiplication(t, _ps), ps, false)
    end
end
