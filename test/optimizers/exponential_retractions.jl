using Test
using LinearAlgebra
using Printf

include("../../src/arrays/skew_sym.jl")
include("../../src/arrays/stiefel_lie_alg_hor.jl")
include("../../src/optimizers/householder.jl")
include("../../src/optimizers/manifold_types.jl")
include("../../src/optimizers/lie_alg_lifts.jl")
include("../../src/arrays/auxiliary.jl")
include("../../src/optimizers/retractions.jl")

#NOTE: zeros have to be added because exp() is not defined for SkewSymMatrix or StiefelLieAlgHorMatrix!!!
function exponential_retraction₁(Y::StiefelManifold, Δ::AbstractMatrix, η)
    StiefelManifold(exp(η*Ω(Y,Δ) - zeros(size(Y,1),size(Y,1)))*Y)
end

function exponential_retraction₂(Y::StiefelManifold, Δ::AbstractMatrix, η)
    N, n = size(Y)
    #@time HD, B = global_rep(Y, Δ)
    E = StiefelProjection(N, n)
    Y₂ = StiefelManifold(exp(η*B - zeros(size(Y,1),size(Y,1)))*E)
    apply_λ(Y, HD, Y₂)
end


N_max = 500
n_max = 10
num = 10
N_vec = Int.(ceil.(rand(num)*N_max))
n_vec = Int.(ceil.(rand(num)*n_max))
n_vec = min.(n_vec, N_vec)
ε = 1e-12
η = .1

for (N, n) ∈ zip(N_vec, n_vec)
    print("N = "*string(N)*", n = "*string(n)*"\n")
    Y = StiefelManifold(N, n)
    Δ = SkewSymMatrix(N)*Y
    @printf "Standard exponential:                                "
    @time sol₁ = exponential_retraction₁(Y, Δ, η)
    @printf "Exponential with householder (expected to be slower):"
    @time sol₂ = exponential_retraction₂(Y, Δ, η)
    @printf "Custom implementation (also gives householder):      "
    @time sol₃ = Exp(Y, Δ, η)
    @test norm(sol₁ - sol₂) < ε
    @test norm(sol₁ - sol₃) < ε
    print("\n")
end