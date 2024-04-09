using GeometricMachineLearning, Test
using GeometricMachineLearning: upper_triangular_asymmetrize
using GeometricMachineLearning: orthonormal_activation
using LinearAlgebra: det
import Random 

Random.seed!(1234)

function attention_tests(N, T=Float32)
    model₁ = Attention(N, Stiefel=false)
    model₂ = Attention(N, Stiefel=true)
    model₃ = Attention(N, orthonormal_activation, Stiefel=false)
    # same as model₁, but with the add connection
    model₄ = Attention(N, Stiefel=false, add_connection=true)

    ps₁ = initialparameters(model₁, CPU(), T)
    ps₂ = initialparameters(model₂, CPU(), T)
    ps₃ = initialparameters(model₃, CPU(), T)
    @test typeof(ps₂.PQ) <: StiefelManifold 
    @test typeof(ps₂.PK) <: StiefelManifold 

    A = randn(N, N)
    det₁ = det(A)
    det₂ = det(model₁(A, ps₁))
    det₃ = det(model₂(A, ps₂))
    det₄ = det(model₃(A, ps₃))
    @test isapprox(det₁, det₂)
    @test isapprox(det₂, det₃)
    @test isapprox(det₃, det₄)

    @test isapprox(model₁(A, ps₁), model₄(A, ps₁)-A)

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