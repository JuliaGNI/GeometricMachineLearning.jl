using Test
using LinearAlgebra

include("../../src/arrays/skew_sym.jl")
include("../../src/arrays/stiefel_lie_alg_hor.jl")
include("../../src/optimizers/householder.jl")
include("../../src/optimizers/manifold_types.jl")
include("../../src/optimizers/lie_alg_lifts.jl")
include("../../src/arrays/auxiliary.jl")
include("../../src/optimizers/retractions.jl")

function exponential_retraction₁(Y::StiefelManifold, Δ::AbstractMatrix, η)
    StiefelManifold(exp(η*Ω(Y,Δ))*Y)
end

function exponential_retraction₂(Y::StiefelManifold, Δ::AbstractMatrix, η)
    N, n = size(Y)
    HD, B = global_rep(Y, Δ)
    E = StiefelProjection(N, n)
    Y₂ = StiefelManifold(exp(η*B)*E)
    apply_λ(Y, HD, Y₂)
end

#function exponentail_retraction_test₃(Y::StiefelManifold, Δ::AbstractMatrix, η)
#end

N_max = 10
n_max = 5
num = 10
N_vec = Int.(ceil.(rand(num)*N_max))
n_vec = Int.(ceil.(rand(num)*n_max))
n_vec = min.(n_vec, N_vec)
ε = 1e-12
η = .1

for (N, n) ∈ zip(N_vec, n_vec)
    Y = StiefelManifold(N, n)
    Δ = SkewSymMatrix(N)*Y
    @test norm(exponential_retraction₁(Y, Δ, η) - exponential_retraction₂(Y, Δ, η)) < ε
    @test norm(exponential_retraction₁(Y, Δ, η) - Exp(Y, Δ, η)) < ε
end