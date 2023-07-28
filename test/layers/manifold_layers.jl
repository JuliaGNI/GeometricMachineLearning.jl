using Test, LinearAlgebra, GeometricMachineLearning

function stiefel_layer_test(T, N, M, tol=1f-1)
    model = Chain(StiefelLayer(N, M), StiefelLayer(N, N))
    ps = initialparameters(T, model)
    o = Optimizer(AdamOptimizer(T(1f0), T(5f-1), T(5f-1), T(3f-7)),ps)

    dx = ((weight=rand(T,N,M),),(weight=rand(T,N,N),))
    ps_copy = deepcopy(ps)
    optimization_step!(o, model, ps, dx)
    # check that the new weight is different from the old one
    @test norm(ps_copy[1].weight - ps[1].weight) > T(tol)
    # check that the updated elements are on the Stiefel Manifold 
    @test typeof(ps[1].weight) <: StiefelManifold{T}
    @test typeof(ps[2].weight) <: StiefelManifold{T}
    ps, ps_copy
end

function grassmann_layer_test(T, N, M, tol=1f-1)
    model = Chain(GrassmannLayer(N, M), StiefelLayer(N, N))
    ps = initialparameters(T, model)
    o = Optimizer(AdamOptimizer(T(1f0), T(5f-1), T(5f-1), T(3f-7)),ps)

    dx = ((weight=rand(T,N,M),),(weight=rand(T,N,N),))
    ps_copy = deepcopy(ps)
    for i in 1:4 optimization_step!(o, model, ps, dx) end
    # check that the new weight is different from the old one
    @test norm(ps_copy[1].weight - ps[1].weight) > T(tol)
    # check that the updated elements are on the Stiefel Manifold 
    @test typeof(ps[1].weight) <: GrassmannManifold{T}
    @test typeof(ps[2].weight) <: StiefelManifold{T}
end

types = (Float32, Float64)
N_max = 10

for T in types
    for N = 3:N_max
        for M = 1:(N-2)
            stiefel_layer_test(T, N, M) 
            grassmann_layer_test(T, N, M)
        end
    end
end