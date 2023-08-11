using GeometricMachineLearning, CUDA, Test

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
    ps = initialparameters(backend, T, model)
    @test typeof(ps[1].weight) <: StiefelManifold{T, <:CuArray{T, 2}}
    @test typeof(ps[2].weight) <: StiefelManifold{T, <:CuArray{T, 2}}

end