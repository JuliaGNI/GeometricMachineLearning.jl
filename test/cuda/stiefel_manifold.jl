using GeometricMachineLearning, CUDA, Test
using KernelAbstractions, Random

backend = CUDA.CUDABackend()

function test_manifold(T, N, n)
    Y = rand(backend, StiefelManifold{T}, N, n)
    @test typeof(Y) <: StiefelManifold{T, <:CuArray{T, 2}}
end

function test_stiefel_lie_alg_hor_matrix(T, N, n)
    B = zeros(backend, StiefelLieAlgHorMatrix{T}, N, n)
    @test typeof(Y) <: StiefelLieAlgHorMatrix{T, <:SkewSymMatrix{T, <:CuArray{T, 1}}, <:CuArray{T, 2}}
end

function test_optimizer(T, N, n)
    model = Chain(StiefelLayer(N, n), StiefelLayer(n, n))
    ps = NeuralNetwork(model, backend, T).params
    @test typeof(ps[1].weight) <: StiefelManifold{T, <:CuArray{T, 2}}
    @test typeof(ps[2].weight) <: StiefelManifold{T, <:CuArray{T, 2}}

    #dx = ((weight=rand(backend, StiefelLieAlgHorMatrix{T}, N, n),), (weight=rand(backend, StiefelLieAlgHorMatrix{T}, n, n),))
    weight1 = KernelAbstractions.allocate(backend, T, N, n)
    weight2 = KernelAbstractions.allocate(backend, T, n, n)
    rand!(Random.default_rng(), weight1)
    rand!(Random.default_rng(), weight2)

    dx = ((weight=weight1,), (weight=weight2,))
    
    o1 = Optimizer(GradientOptimizer(), ps)
    o2 = Optimizer(MomentumOptimizer(), ps)
    o3 = Optimizer(AdamOptimizer(), ps)

    optimization_step!(o1, model, ps, dx)
    optimization_step!(o2, model, ps, dx)
    optimization_step!(o3, model, ps, dx)
end