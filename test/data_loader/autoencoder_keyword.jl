# Data-set constructors accept `autoencoder::Bool`. A value outside `Bool`, such as `nothing`,
# is rejected at the keyword with a `TypeError` rather than interpreted as a loader kind.

using GeometricMachineLearning
using Test
import Random

Random.seed!(1234)

@testset "DataLoader(::AbstractArray{<:Number, 3})" begin
    data = rand(Float32, 2, 3, 4)
    @test_throws TypeError DataLoader(data; autoencoder = nothing, suppress_info = true)
end

@testset "DataLoader(::AbstractMatrix)" begin
    data = rand(Float32, 2, 3)
    @test_throws TypeError DataLoader(data; autoencoder = nothing, suppress_info = true)
end

@testset "DataLoader(::NamedTuple{(:q, :p)}) of matrices" begin
    data = (q = rand(Float32, 2, 3), p = rand(Float32, 2, 3))
    @test_throws TypeError DataLoader(data; autoencoder = nothing, suppress_info = true)
    # the vector pair reaches the same branch through the matrices above
    @test_throws TypeError DataLoader((q = rand(Float32, 3), p = rand(Float32, 3));
        autoencoder = nothing, suppress_info = true)
end

@testset "DataLoader(::NamedTuple{(:q, :p)}) of tensors" begin
    data = (q = rand(Float32, 2, 3, 4), p = rand(Float32, 2, 3, 4))
    @test_throws TypeError DataLoader(data; autoencoder = nothing, suppress_info = true)
end

# `DataLoader(dl, backend)` accepts `autoencoder::Union{Nothing, Bool}`: `nothing` inherits the
# loader's kind, a Boolean selects it, and a value outside the union raises a `TypeError`.
@testset "DataLoader(dl, backend): nothing means inherit, and nothing falls through" begin
    dl = DataLoader(rand(Float32, 2, 3, 4); autoencoder = true, suppress_info = true)

    inherited = DataLoader(dl, GeometricMachineLearning.CPU())
    @test typeof(inherited).parameters[4] === typeof(dl).parameters[4] === :RegularData

    @test_throws TypeError DataLoader(dl, GeometricMachineLearning.CPU(); autoencoder = "x")
end
