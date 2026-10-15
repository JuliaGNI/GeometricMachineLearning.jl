using Test
using GeometricMachineLearning
using GeometricMachineLearning: _mul
import Random

include("../helpers/network_parameters_gradient_structure.jl")

Random.seed!(1234)

function single_multiplication(a::AT, _ps::Union{
        NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    _mul(a, _ps.L1.A)
end

function double_multiplication(a::AT, _ps::Union{
        NamedTuple, NetworkParameters}) where {AT <: AbstractArray}
    _mul(_ps.L1.A, _mul(a, _ps.L1.A))
end

S = rand(SymmetricMatrix, 4)
ps = NetworkParameters((L1 = (A = S,),))
t = rand(4, 4)

@testset "_mul: NetworkParameters gradient structure" begin
    @testset "single_multiplication (one wrapper access)" begin
        network_parameters_gradient_structure_test(
            _ps -> single_multiplication(t, _ps), ps)
    end
    @testset "double_multiplication (two wrapper accesses)" begin
        network_parameters_gradient_structure_test(
            _ps -> double_multiplication(t, _ps), ps)
    end
end
