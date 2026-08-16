using GeometricMachineLearning
using Zygote
using Test
using LinearAlgebra: norm, I
import Random

Random.seed!(123)

@doc raw"""
Pin the momentum recursion.

For a *constant* gradient ``g`` the momentum ``p^{(t)} = \alpha{}p^{(t-1)} + g`` is the geometric sum
``p^{(t)} = g(1 - \alpha^t)/(1 - \alpha)``, so the step ``\eta{}p^{(t)}`` saturates at
``\eta{}g/(1-\alpha)``. The recursion `p ← p + αg` also produces a growing sequence, but a linearly
growing one that never saturates, which is what this test rules out.
"""
function momentum_matches_geometric_sum(; α = 0.9, η = 0.1, n_steps = 10)
    ps = (A = zeros(2, 2),)
    o = Optimizer(MomentumMethod(α), ps; step_size = η)
    λY = GlobalSection(ps)
    previous = 0.0
    for t in 1:n_steps
        optimization_step!(o, λY, ps, (A = ones(2, 2),))
        step = previous - ps.A[1, 1]
        previous = ps.A[1, 1]
        @test step ≈ η * (1 - α^t) / (1 - α)
    end
    # the steps have to stay below the saturation value, not run past it
    @test previous > -n_steps * η / (1 - α)
end

@doc raw"""
Momentum on the Stiefel manifold reduces the loss and keeps the iterate on the manifold.
"""
function momentum_optimizer_stiefel(N, n; n_steps = 50, step_size = 1e-3, α = 0.5)
    YB = rand(StiefelManifold, N, n)
    B = YB * YB'
    loss(ps) = norm(ps.Y * ps.Y' - B)^2
    ps = (Y = rand(StiefelManifold, N, n),)
    loss1 = loss(ps)
    o = Optimizer(MomentumMethod(α), ps; step_size = step_size)
    λY = GlobalSection(ps)
    for _ in 1:n_steps
        optimization_step!(o, λY, ps, Zygote.gradient(loss, ps)[1])
    end
    @test loss(ps) < loss1
    @test ps.Y' * ps.Y ≈ I
end

momentum_matches_geometric_sum()
momentum_optimizer_stiefel(10, 5)
