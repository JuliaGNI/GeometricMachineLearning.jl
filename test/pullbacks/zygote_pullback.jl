using GeometricMachineLearning
using GeometricMachineLearning: FeedForwardLoss, QPTOAT, LASympNet, StiefelLayer
using AbstractNeuralNetworks: AbstractExplicitLayer, Chain, Dense, NeuralNetwork, params
using LinearAlgebra: Adjoint, Diagonal, diag
using NeuralNetworkParameters: NetworkParameters
using Test
import Random
import Zygote

include("../helpers/eltypes.jl")

# A `NetworkLoss` that reads `ps.L1.A` twice, in the form of `double_multiplication` in
# `test/layers/double_multiplication_network_parameters_gradient.jl`: one read leaves the leaf a
# `SymmetricMatrix` under either path, and only the second read separates them. `ps` is untyped, as
# the layer functors' per-layer arguments are: the reference below differentiates the same body
# against the bare `NamedTuple`, which is the shape `Zygote.gradient` projects.
struct DoubleReadLoss <: GeometricMachineLearning.NetworkLoss end

function (::DoubleReadLoss)(
        model::Union{Chain, AbstractExplicitLayer}, ps, input::QPTOAT)
    A = ps.L1.A
    sum(abs2, model(input, ps).q) + sum(abs2, A) + sum(abs2, A)
end

# One pullback shape: `back(Δ)` is a `NetworkParameters` with the keys of the parameters it was built
# from, not a one-tuple of one. The rows below build the same assertion on a dense `Chain`, on an
# `LA` SympNet, which holds a `SymmetricMatrix` leaf, and on a Stiefel one, because a leaf type is
# where a shape claim can break silently: the generated tangent has the leaf's own type or its
# storage, and nothing in the `NetworkParameters` wrapper says which.
@testset "ZygotePullback's back(Δ) is a NetworkParameters ($T)" for T in REAL_ELTYPES
    networks = (
        (model = Chain(Dense(4, 2, tanh)), input = rand(T, 4, 3), output = rand(T, 2, 3)),
        (model = LASympNet(4; depth = 1, nhidden = 1),
            input = rand(T, 4, 3), output = rand(T, 4, 3)),
        (model = Chain(StiefelLayer(6, 3), Dense(3, 2, tanh)), input = rand(T, 6, 3),
            output = rand(T, 2, 3))
    )
    for (index, network) in enumerate(networks)
        Random.seed!(index)
        nn = NeuralNetwork(network.model, CPU(), T)
        ps = params(nn)
        value, back = ZygotePullback(FeedForwardLoss())(
            ps, nn.model, (network.input, network.output))
        g = back(one(value))
        @test value isa T
        @test g isa NetworkParameters
        @test keys(g) == keys(ps)
        for (name, leaf) in pairs(g)
            @test keys(leaf) == keys(ps[name])
            for k in keys(leaf)
                @test eltype(leaf[k]) === T
                @test size(leaf[k]) == size(ps[name][k])
            end
        end
        # The Stiefel leaf's tangent is an `Adjoint` of its storage rather than a `StiefelManifold`:
        # the manifold layer's functor reads the weight through an adjoint, and Zygote keeps the
        # cotangent's own type instead of projecting it back onto the leaf's.
        if network.model isa Chain && network.model.layers[1] isa StiefelLayer
            @test g.L1.weight isa Adjoint{T, Matrix{T}}
            @test size(g.L1.weight) == size(ps.L1.weight)
        end
    end
end

# The gradient of a `SymmetricMatrix` leaf is the gradient with respect to the leaf's *storage*, so
# the off-diagonal entries are twice those of the projection of the dense gradient (issue #319,
# decision 8). The reference is a `Zygote.gradient` with respect to the bare `NamedTuple` the wrapper
# holds, which projects; one read of the leaf gives either relation, and two make the wrapped one
# visible.
@testset "a SymmetricMatrix leaf holds its storage gradient ($T)" for T in REAL_ELTYPES
    Random.seed!(7)
    nn = NeuralNetwork(Chain(SymplecticAttentionQ(4; symmetric = true)), CPU(), T)
    ps = params(nn)
    x = (q = rand(T, 2, 2), p = rand(T, 2, 2))

    value, back = ZygotePullback(DoubleReadLoss())(ps, nn.model, x)
    g = back(one(value))

    projected = Zygote.gradient(p -> DoubleReadLoss()(nn.model, p, x), params(ps))[1]
    B = Matrix(projected.L1.A)
    @test value isa T
    @test g.L1.A isa SymmetricMatrix{T}
    @test Matrix(g.L1.A) ≈ 2B - Diagonal(diag(B))
end
