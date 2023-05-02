"""
This implements tests for the Riemannian gradients.

TODO: find correct expression for Riemannian gradient for the canonical metric!
"""

using Test
using LinearAlgebra

include("../../src/optimizers/householder.jl")
include("../../src/optimizers/manifold_types.jl")
include("../../src/optimizers/riemannian_gradients.jl")
include("../../src/arrays/skew_sym.jl")

#Riemannian metric for the Stiefel manifold -> this is probably not needed explicitly!
function riemannian_metric(Y::StiefelManifold, Δ₁, Δ₂)
    tr(Δ₁'*(I - .5*Y*Y')*Δ₂)
end


function stiefel_riemannian_gradient_test(N::Int, n::Int, ε = 1e-12)
    #sample element from 𝔐
    Y = StiefelManifold(N,n)
    #sample element from T𝔐 (tangent space):
    V = SkewSymMatrix(randn(N,N))*Y
    #sample element from T*𝔐 (cotangent space):
    A = randn(N,n)
    @test norm(tr(A'*V) - riemannian_metric(Y, grad(Y, A), V)) < ε
end

N_max = 20
n_max = 10
num = 10
N_vec = Int.(ceil.(rand(num)*N_max))
n_vec = Int.(ceil.(rand(num)*n_max))
n_vec = min.(n_vec, N_vec)

for (N, n) ∈ zip(N_vec, n_vec)
    stiefel_riemannian_gradient_test(N, n)
end