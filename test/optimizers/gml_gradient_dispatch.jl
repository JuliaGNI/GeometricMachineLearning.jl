# What `_GMLGradient` is called on, and by which method.
#
# `_GMLGradient` is a `GeometricOptimizers.Gradient`, so its functor methods share a method table with
# upstream's. Two of them overlap on a shape this package actually produces, and the overlap is an
# *ambiguity* rather than a choice — which is a run-time error several frames into a solve, not a
# signature a reader would notice. So it is asserted here rather than left to whichever call path
# happens to reach it.

using GeometricMachineLearning
using GeometricOptimizers: StiefelManifold, GrassmannManifold, rgrad
using NeuralNetworkParameters: NetworkParameters
using Test

GML = GeometricMachineLearning

# A bare `Manifold`. `_GMLGradient{T} <: GeometricOptimizers.Gradient{T}` and `Manifold{T} <:
# AbstractMatrix{T}`, so `(::_GMLGradient{T})(::AbstractArray{T})` and upstream's
# `(::Gradient{T})(::Manifold{T})` are ambiguous here: neither is more specific. Without a method of
# its own this call is `MethodError: ... is ambiguous`.
#
# The value is this package's answer and not upstream's. Upstream's body is
# `rgrad(x, reshape(grad(vec(x)), size(x)...))`, which evaluates an *inner* gradient — and a
# `_GMLGradient` has none, since the whole point of it is that the Euclidean gradient was computed
# already and is being carried in `dp`.
@testset "a bare Manifold reaches this package's method, not upstream's" begin
    for M in (StiefelManifold, GrassmannManifold)
        Y  = rand(M, 4, 2)
        dp = randn(4, 2)
        g  = GML._GMLGradient{Float64, typeof(dp)}(dp)

        @test g(Y) == rgrad(Y, dp)
        @test which(g, Tuple{typeof(Y)}).module === GeometricMachineLearning
    end
end

# The two shapes the optimizer actually hands it, so that the method above is not the only one
# covered and a later narrowing of either shows up here.
@testset "a wrapped layer and a plain array" begin
    dp = randn(3, 3)
    g  = GML._GMLGradient{Float64, typeof(dp)}(dp)
    # a plain array leaf: the gradient is already Euclidean, so it passes through
    @test g(randn(3, 3)) === dp

    ps = NetworkParameters((weight = rand(StiefelManifold, 4, 2),))
    dps = (weight = randn(4, 2),)
    gps = GML._GMLGradient{Float64, typeof(dps)}(dps)
    out = gps(ps)
    # rebuilt in the shape of its *first* argument -- a container, which is what
    # `GeometricOptimizers._copyto!` has a method for
    @test out isa NetworkParameters
    @test out.weight == rgrad(ps.weight, dps.weight)
end

# `_is_layer` is the rule for how a network is split into caches, so its boundary cases are stated
# rather than inherited. An empty set is *zero* caches' worth and descends; `all` over an empty
# collection would otherwise make it vacuously one.
@testset "_is_layer at the boundary" begin
    @test GML._is_layer((W = randn(2, 2), b = randn(2)))
    @test GML._is_layer(NetworkParameters((W = randn(2, 2), b = randn(2))))
    # a set whose values are branches is a subtree, and gets one cache per layer
    @test !GML._is_layer((L1 = (W = randn(2, 2),), L2 = (W = randn(2, 2),)))
    @test !GML._is_layer(NamedTuple())
    @test !GML._is_layer(NetworkParameters(NamedTuple()))
    @test !GML._is_layer(randn(2, 2))

    # a layer whose weights do not share an element type is still one cache, with `T` the promotion
    @test GML._is_layer((W = randn(Float32, 2, 2), b = randn(Float64, 2)))
end

# `_as_go_solution` is the wrap itself, and the property that makes the wrap free: it shares the leaf
# arrays, so the optimizer step writes through to the network's own weights and nothing is copied
# back. That is what makes this a boundary change rather than a change of behaviour, and it is the one
# claim of this release that a reader cannot check by reading a signature.
@testset "_as_go_solution wraps a layer and shares its arrays" begin
    W, b = randn(2, 2), randn(2)
    layer = (W = W, b = b)

    wrapped = GML._as_go_solution(layer)
    @test wrapped isa NetworkParameters
    @test wrapped.W === W          # shared, not copied
    @test wrapped.b === b

    # already in the right shape: passed through untouched
    ps = NetworkParameters(layer)
    @test GML._as_go_solution(ps) === ps
    Y = rand(StiefelManifold, 4, 2)
    @test GML._as_go_solution(Y) === Y
    v = randn(3)
    @test GML._as_go_solution(v) === v

    # a subtree is not a layer, so it is left alone for the tree branch to descend
    tree = (L1 = (W = randn(2, 2),), L2 = (W = randn(2, 2),))
    @test GML._as_go_solution(tree) === tree
end

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
