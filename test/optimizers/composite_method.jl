# One optimizer method per layer, through `GeometricOptimizers.CompositeMethod`.
#
# A symplectic autoencoder is the shape this exists for: its PSD layers carry only Stiefel weights
# and its SympNet layers only Euclidean ones, and `ScalarMomentAdam` -- whose second moment is a
# scalar, and therefore a statement about *one* manifold -- accepts exactly one `StiefelManifold`
# solution. Before the composite, a caller wanting that method on the Stiefel layers and ordinary
# `Adam` on the rest had to add methods to `_make_optimizer_cache`, `_make_optimizer_state` and
# `_leaf_optim_step!` from outside this package, which is type piracy on four underscore-prefixed
# internals and breaks silently on any refactor here.
#
# What is pinned below is that the selection reaches the cache, the state and the step; that a
# single-weight-scope method is handed the weight and not the layer; that its scalar moment is
# carried back into the state; and that a uniform method is byte-for-byte unaffected.

using GeometricMachineLearning
using GeometricMachineLearning: params, _as_go_solution, _is_go_native_method,
                                _make_optimizer_cache, _make_optimizer_state
using Test
import GeometricOptimizers
import Random

const GO = GeometricOptimizers

Random.seed!(1234)

composite(T) = GO.CompositeMethod(;
    manifold = GO.ScalarMomentAdam(T), array = GO.Adam(T))

@testset "ScalarMomentAdam is driven through a GeometricOptimizers cache" begin
    # Without this it fell through to `GMLEuclideanState`, which is coordinate-wise Adam on the
    # ambient array: not the method, and not on the manifold.
    @test _is_go_native_method(GO.ScalarMomentAdam())
    @test_throws ArgumentError _is_go_native_method(composite(Float64))
end

@testset "the cache and state trees follow the selection layer by layer" begin
    T = Float32
    ps = params(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 1,
            n_decoder_blocks = 1, n_encoder_layers = 2, n_decoder_layers = 2), T))
    caches = _make_optimizer_cache(composite(T), ps)
    states = _make_optimizer_state(composite(T), ps)

    stiefel_caches = 0
    adam_caches = 0
    function walk(node, layer)
        if node isa NamedTuple && !(node isa GO.OptimizerCache)
            for key in keys(node)
                walk(node[key], layer[key])
            end
            return nothing
        end
        if all(weight -> weight isa StiefelManifold, values(layer))
            @test node isa GO.ScalarMomentAdamCache
            stiefel_caches += 1
        else
            @test node isa GO.AdamCache
            adam_caches += 1
        end
        nothing
    end
    walk(caches, ps)

    @test stiefel_caches > 0        # the PSD layers
    @test adam_caches > 0           # the SympNet layers
    @test states isa NamedTuple
end

@testset "a single-weight-scope method is handed the weight, not the layer" begin
    Y = rand(StiefelManifold{Float32}, 4, 2)
    layer = (weight = Y,)

    @test _as_go_solution(layer, GO.ScalarMomentAdam(Float32)) === Y
    @test _as_go_solution(layer, GO.Adam(Float32)) isa
          GeometricMachineLearning.NetworkParameters
    @test _as_go_solution(Y, GO.ScalarMomentAdam(Float32)) === Y

    # A layer of more than one weight has no single weight to be handed, and the error says which
    # layer and which method rather than surfacing as upstream's scope message several frames down.
    wide = (weight = Y, bias = rand(Float32, 2))
    @test_throws ArgumentError _as_go_solution(wide, GO.ScalarMomentAdam(Float32))
end

@testset "a composite trains a symplectic autoencoder and keeps it symplectic" begin
    T = Float32
    architecture = SymplecticAutoencoder(4, 2; n_encoder_blocks = 1, n_decoder_blocks = 1,
        n_encoder_layers = 2, n_decoder_layers = 2)
    nn = NeuralNetwork(architecture, T)
    dl = DataLoader(rand(T, 4, 20, 5); autoencoder = true, suppress_info = true)

    optimizer = Optimizer(composite(T), nn; retraction = cayley, step_size = 1.0f-3)
    losses = optimizer(nn, dl, Batch(5), 2; show_progress = false)

    @test length(losses) == 2
    @test all(isfinite, losses)
    # Every Stiefel weight is still on the manifold: the composite's Stiefel arm retracts, it does
    # not step in the ambient space.
    for layer in values(params(nn)), weight in values(layer)
        weight isa StiefelManifold && @test GO.check(weight) < 1.0f-4
    end
end

@testset "the scalar second moment is carried back into the state" begin
    # `sync_state!`'s job, and the one the `isa` chain this replaced simply did not have an arm for:
    # without it the moment restarts from zero on every step and the method is its first iteration
    # forever.
    T = Float64
    Y = rand(StiefelManifold{T}, 4, 2)
    ps = GeometricMachineLearning.NetworkParameters((L1 = (weight = Y,),))
    method = composite(T)
    caches = _make_optimizer_cache(method, ps)
    states = _make_optimizer_state(method, ps)
    λY = GlobalSection(ps)
    dp = (L1 = (weight = rand(T, 4, 2),),)

    @test GO.second_moment(states.L1) == 0
    GeometricMachineLearning._tree_optim_step!(caches, states, dp, ps, λY, method,
        cayley, 1.0e-3)
    @test GO.second_moment(states.L1) > 0
    @test GO.second_moment(states.L1) == GO.second_moment(caches.L1)
    @test states.L1.iterations == 1
end

@testset "a uniform method is unaffected" begin
    T = Float32
    architecture = SymplecticAutoencoder(4, 2; n_encoder_blocks = 1, n_decoder_blocks = 1,
        n_encoder_layers = 2, n_decoder_layers = 2)

    function run(method)
        Random.seed!(77)
        nn = NeuralNetwork(architecture, T)
        dl = DataLoader(rand(Random.Xoshiro(3), T, 4, 20, 5); autoencoder = true,
            suppress_info = true)
        optimizer = Optimizer(method, nn; retraction = cayley, step_size = 1.0f-3)
        optimizer(nn, dl, Batch(5), 2; show_progress = false)
    end

    # `Adam` everywhere, and a composite that selects `Adam` everywhere, have to be the same run:
    # the composite is a choice of method and not a second implementation of one.
    @test run(GO.Adam(T)) ==
          run(GO.CompositeMethod(; manifold = GO.Adam(T), array = GO.Adam(T)))
end
