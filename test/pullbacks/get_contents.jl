using GeometricMachineLearning
using NeuralNetworkParameters: NetworkParameters
using Test
import Random

Random.seed!(1234)

GML = GeometricMachineLearning

# `_get_contents` unwraps whatever shape the pullback handed back. Six methods, because a reverse pass
# produces a container or the bare `NamedTuple` it wraps, and `Zygote` may hand either back bare, in a
# one-tuple, or in a one-element vector. Each is exercised here: splitting a union into a method per
# shape means each method needs its own case, or the split trades one tested path for six untested
# ones.
@testset "_get_contents unwraps every shape a pullback returns" begin
    nt = (L1 = (W = randn(2, 2), b = randn(2)),)
    ps = NetworkParameters(nt)

    for x in (nt, ps)
        @test GML._get_contents(x) === x
        @test GML._get_contents((x,)) === x
        @test GML._get_contents([x]) === x
    end

    # more than one set is a caller error and says so, rather than silently taking the first
    @test_throws ArgumentError GML._get_contents([ps, ps])
    @test_throws ArgumentError GML._get_contents([nt, nt])
end
