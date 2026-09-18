# Each of these constructors was `if autoencoder == false … elseif autoencoder == true … end`,
# with no `else`. A value that is neither -- `nothing` being the case that actually arises, since
# it is the sentinel a caller reaches for when in doubt -- fell through and the constructor
# returned `nothing` instead of a `DataLoader`, with no indication of why. Declaring the keyword
# `autoencoder::Bool` turns that silent `nothing` into a `TypeError` at the call the keyword is
# passed to, which is where the mistake actually is.

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
# keeps `Union{Nothing, Bool}` and its `DT = if …` block gained the `else` the others gained a type
# for, so a value outside that union is a `TypeError` there too, and the fallthrough is gone the
# same way.
@testset "DataLoader(dl, backend): nothing still means inherit, and the fallthrough is gone" begin
    dl = DataLoader(rand(Float32, 2, 3, 4); autoencoder = true, suppress_info = true)

    inherited = DataLoader(dl, GeometricMachineLearning.CPU())
    @test typeof(inherited).parameters[end] == typeof(dl).parameters[end]

    @test_throws TypeError DataLoader(dl, GeometricMachineLearning.CPU(); autoencoder = "x")
end
