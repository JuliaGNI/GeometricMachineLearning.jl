# `GeometricMachineLearning._norm` has three arms for one function, and all three keep the
# element type of their argument: the two `NamedTuple` arms divide by `√2` and `√length(dx)` in
# that type, and the plain-`AbstractArray` arm does so directly. It reaches the user through
# `reduction_error` and `projection_error`, so a `Float32` reduced system reports its error in
# `Float32`, not `Float64`. See `CHANGELOG.md` for the fix this guards.

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
