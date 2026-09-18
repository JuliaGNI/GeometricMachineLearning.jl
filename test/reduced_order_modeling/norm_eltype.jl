# `GeometricMachineLearning._norm` had three arms for one function, and the two `NamedTuple` arms
# widened a `Float32` argument to `Float64` (dividing by `√2` and `√length(dx)`, both computed as
# `Float64`), while the plain-`AbstractArray` arm was already correct. It reaches the user through
# `reduction_error` and `projection_error`, so a `Float32` reduced system used to report a `Float64`
# error.

using GeometricMachineLearning
using Test

GML = GeometricMachineLearning

@testset "_norm keeps the element type of its argument" begin
    @test GML._norm((q = rand(Float32, 4), p = rand(Float32, 4))) isa Float32
    @test GML._norm((a = rand(Float32, 4), b = rand(Float32, 4))) isa Float32
    @test GML._norm(rand(Float32, 4)) isa Float32

    @test GML._norm((q = rand(Float64, 4), p = rand(Float64, 4))) isa Float64
    @test GML._norm((a = rand(Float64, 4), b = rand(Float64, 4))) isa Float64
    @test GML._norm(rand(Float64, 4)) isa Float64
end
