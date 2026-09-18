# Each of these constructors branches on `autoencoder`, so a value that is neither `true` nor
# `false` -- `nothing` being the case that actually arises, since it is the sentinel a caller
# reaches for when in doubt -- must be rejected at the keyword rather than inside the branch.
# Declaring the keyword `autoencoder::Bool` makes it a `TypeError` at the call the keyword is
# passed to, which is where the mistake is. Without the declaration such a value falls through an
# `if`/`elseif` with no `else`, and the constructor returns `nothing` instead of a `DataLoader`
# with no indication of why.

using GeometricMachineLearning
using Test

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
end

@testset "DataLoader(::NamedTuple{(:q, :p)}) of tensors" begin
    data = (q = rand(Float32, 2, 3, 4), p = rand(Float32, 2, 3, 4))
    @test_throws TypeError DataLoader(data; autoencoder = nothing, suppress_info = true)
end

# The fifth site, `DataLoader(dl, backend)`, is not one of the four above: its `autoencoder =
# nothing` default is not a mistake to reject, it is the sentinel meaning "inherit from `dl`". It
# takes `Union{Nothing, Bool}` instead, and its `DT = if …` block carries the `else` that the type
# gives the other four, so a value outside that union is a `TypeError` there too and nothing falls
# through.
@testset "DataLoader(dl, backend): nothing means inherit, and nothing falls through" begin
    dl = DataLoader(rand(Float32, 2, 3, 4); autoencoder = true, suppress_info = true)

    inherited = DataLoader(dl, GeometricMachineLearning.CPU())
    @test typeof(inherited).parameters[end] == typeof(dl).parameters[end]

    @test_throws TypeError DataLoader(dl, GeometricMachineLearning.CPU(); autoencoder = "x")
end
