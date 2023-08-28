using GeometricMachineLearning, Test
using GeometricMachineLearning: upper_triangular_asymmetrize
using LinearAlgebra: det

function attention_tests(N, T=Float32)
    model₁ = Attention(N, Stiefel=false)
    model₂ = Attention(N, Stiefel=true)

    ps₁ = initialparameters(CPU(), T, model₁)
    ps₂ = initialparameters(CPU(), T, model₂)
    @test typeof(ps₂.PQ) <: StiefelManifold 
    @test typeof(ps₂.PQ) <: StiefelManifold 

    A = randn(N, N)
    det₁ = det(A)
    det₂ = det(model₁(A, ps₁))
    det₃ = det(model₂(A, ps₂))
    @test isapprox(det₁, det₂)
    @test isapprox(det₂, det₃)

    A = reshape(rand(SkewSymMatrix{T}, N), N, N, 1)
    @test isapprox(A, upper_triangular_asymmetrize(A))

    model = Chain(model₁, model₂)
    ps = (ps₁, ps₂)
    dx₁ = (PQ=rand(T, N, N), PK=rand(T, N, N))
    dx₂ = (PQ=rand(StiefelLieAlgHorMatrix{T}, N, N), PK=rand(StiefelLieAlgHorMatrix{T}, N, N))
    dx = (dx₁, dx₂)
    o = Optimizer(AdamOptimizer(), ps)
    optimization_step!(o, model, ps, dx)
    @test typeof(ps₂.PQ) <: StiefelManifold 
    @test typeof(ps₂.PQ) <: StiefelManifold 
end

attention_tests(10)