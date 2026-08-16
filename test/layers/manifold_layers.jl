using Test, LinearAlgebra, GeometricMachineLearning
import Random 

Random.seed!(1234)

function stiefel_layer_test(T, M, N, tol=1f-1)
    model = Chain(StiefelLayer(M, N), StiefelLayer(N, N))
    ps = NeuralNetwork(model, T).params
    o = Optimizer(Adam(T; β₁ = T(5f-1), β₂ = T(5f-1), δ = T(3f-7)), ps;
        step_size = T(1f0))

    dx = (L1 = (weight = rand(T, N, M),), L2 = (weight=rand(T, N, N),))
    ps_copy = deepcopy(ps)
    λY = GlobalSection(ps)
    optimization_step!(o, λY, ps, dx)
    # check that the new weight is different from the old one
    @test norm(ps_copy[1].weight - ps[1].weight) > T(tol)
    # check that the updated elements are on the Stiefel Manifold 
    @test typeof(ps[1].weight) <: StiefelManifold{T}
    @test typeof(ps[2].weight) <: StiefelManifold{T}
end

function grassmann_layer_test(T, M, N, tol=1f-1)
    model = Chain(GrassmannLayer(M, N), StiefelLayer(N, N))
    ps = NeuralNetwork(model, T).params
    o = Optimizer(Adam(T; β₁ = T(5f-1), β₂ = T(5f-1), δ = T(3f-7)), ps;
        step_size = T(1f0))

    dx = (L1 = (weight=rand(T,N,M),), L2 = (weight=rand(T,N,N),))
    ps_copy = deepcopy(ps)
    λY = GlobalSection(ps)
    for i in 1:4 optimization_step!(o, λY, ps, dx) end
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
            stiefel_layer_test(T, M, N) 
            grassmann_layer_test(T, M, N)
        end
    end
end
