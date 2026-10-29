# The one testset of the loader's files that needs this package's own names: `train!` on a
# `ParametricDataLoader` trains on `ParametricLoss` by default, and the network is the parametric
# affine layer the rest of `test/parametric.jl` builds. The loader and its own tests are
# `ReducedComplexityModeling`'s.
#
# The network is `(x, μ) ↦ W [x; flatten(μ)] + b`. `train!` without `loss` and with
# `loss = ParametricLoss()` draw the same network and the same minibatches under one seed.

using GeometricMachineLearning
using GeometricMachineLearning: ParametricLoss, params
using ReducedComplexityModeling: ParametricDataLoader
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/parametric_affine.jl")

# `m` trajectories of `nt` time steps of a state of dimension `2d`.
const d, nt, m = 1, 5, 3

# The data of trajectory `j` at time step `t`, in every entry, and its system parameters `μⱼ`.
timed_data(::Type{T}) where {T} = [T(100j + t) for _ in 1:(2d), t in 1:nt, j in 1:m]
system_parameters(::Type{T}) where {T} = [(a = T(j),) for j in 1:m]

@testset "train! on a ParametricDataLoader trains on ParametricLoss ($T)" for T in REAL_ELTYPES
    dl = ParametricDataLoader(timed_data(T) ./ 100, system_parameters(T))
    function trained(; kwargs...)
        Random.seed!(3)
        nn = NeuralNetwork(Chain(ParametricAffine{2d}(1)), T)
        initial = deepcopy(params(nn))
        opt = TrainingOptimizer(nn.params; algorithm = Adam())
        history = train!(nn, opt, dl, Batch(4), 3; show_progress = false, kwargs...)
        history, initial, params(nn)
    end
    history, initial, final = trained()
    @test history == first(trained(; loss = ParametricLoss()))
    @test length(history) == 3
    @test eltype(history) === T
    @test final.L1.W != initial.L1.W
    @test final.L1.b != initial.L1.b
end
