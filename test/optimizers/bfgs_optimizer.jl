using GeometricMachineLearning
using Zygote
using Test
using LinearAlgebra: norm
import Random

Random.seed!(123)

@doc raw"""
Test that gradient descent reduces loss on a Euclidean (plain matrix) problem.
BFGSOptimizer is no longer available; this file retains the structure for future extension.
"""
function gradient_optimizer_euclidean(N; n_steps = 20, step_size = 1e-3)
    B = inv(rand(N, N))
    loss(ps) = norm(ps.A - B) ^ 2
    A = randn(N, N)
    ps = (A = A,)
    loss1 = loss(ps)
    o = Optimizer(GradientMethod(), ps; step_size = step_size)
    for _ in 1:n_steps
        ∇L = Zygote.gradient(loss, ps)[1]
        λY = GlobalSection(ps)
        optimization_step!(o, λY, ps, ∇L)
    end
    loss2 = loss(ps)
    @test loss1 > loss2
end

@doc raw"""
Test that gradient descent reduces loss on the Stiefel manifold.
"""
function gradient_optimizer_stiefel(N, n; n_steps = 20, step_size = 1e-3)
    YB = rand(StiefelManifold, N, n)
    B  = YB * YB'
    loss(ps) = norm(ps.Y * ps.Y' - B) ^ 2
    Y  = rand(StiefelManifold, N, n)
    ps = (Y = Y,)
    loss1 = loss(ps)
    o = Optimizer(GradientMethod(), ps; step_size = step_size)
    for _ in 1:n_steps
        ∇L = Zygote.gradient(loss, ps)[1]
        λY = GlobalSection(ps)
        optimization_step!(o, λY, ps, ∇L)
    end
    loss2 = loss(ps)
    @test loss1 > loss2
end

gradient_optimizer_euclidean(10)
gradient_optimizer_stiefel(10, 5)
