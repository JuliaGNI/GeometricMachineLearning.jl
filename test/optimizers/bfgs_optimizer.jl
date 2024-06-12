using GeometricMachineLearning
using GeometricMachineLearning: update_section!
using Zygote
using Test
using LinearAlgebra: norm
import Random

Random.seed!(123)

@doc raw"""
    bfgs_optimizer(N)

Test if BFGS optimizer perfroms better than gradient optimizer.

The test is performed on a simple loss function
```math
    \mathrm{loss}(A) = norm(A - B) ^ 3,
```
where ``B`` is fixed. 
"""
function bfgs_optimizer(N; n_steps = 10, η = 1e-4)
    B = inv(rand(N, N))
    loss(A) = norm(A - B) ^ (2)
    A = randn(N, N)
    loss1 = loss(A)
    method₁ = GradientOptimizer(η)
    o₁ = Optimizer(method₁, (A = A,))
    for _ in 1:n_steps
        ∇L = Zygote.gradient(loss, A)[1]
        update!(o₁, o₁.cache.A, ∇L)
        A .+= ∇L
    end
    loss2 = loss(A)
    A = randn(N, N)
    method₂ = BFGSOptimizer(η)
    o₂ = Optimizer(method₂, (A = A,))
    for _ in 1:n_steps
        ∇L = Zygote.gradient(loss, A)[1]
        update!(o₂, o₂.cache.A, ∇L)
        A .+= ∇L
    end
    loss3 = loss(A)
    @test loss1 > loss2 > loss3
    println(loss2)
    println(loss3)

end

bfgs_optimizer(10)

@doc raw"""
    bfgs_optimizer(N, n)

Test if BFGS optimizer perfroms better than gradient optimizer.

The test is performed on a simple loss function
```math
    \mathrm{loss}(A) = norm(AA^T - B) ^ 3,
```
where ``B = Y_BY_B^T`` for some ``Y\in{}St(n, N)`` is fixed. 
``A`` in the equation above is optimized on the Stiefel manifold. 
"""
function bfgs_stiefel_optimizer(N, n; n_steps = 10, η = 1e-4)
    YB = rand(StiefelManifold, N, n)
    B = YB * YB'
    loss(A) = norm(A * A' - B) ^ 2
    Y = rand(StiefelManifold, N, n)
    λY = GlobalSection(Y)
    loss1 = loss(Y)
    method₁ = GradientOptimizer(η)
    o₁ = Optimizer(method₁, (A = Y,))
    for _ in 1:n_steps
        ∇L = Zygote.gradient(loss, Y)[1]
        gradL = global_rep(λY, ∇L)
        update!(o₁, o₁.cache.A, gradL)
        update_section!(λY, gradL, cayley)
    end
    loss2 = loss(Y)
    Y = rand(StiefelManifold, N, n)
    method₂ = BFGSOptimizer(η)
    o₂ = Optimizer(method₂, (A = Y,))
    for _ in 1:n_steps
        ∇L = Zygote.gradient(loss, Y)[1]
        gradL = global_rep(λY, ∇L)
        update!(o₂, o₂.cache.A, gradL)
        update_section!(λY, gradL, cayley)
    end
    loss3 = loss(Y)
    @test loss1 > loss2 > loss3
    println(loss2)
    println(loss3)
end

bfgs_stiefel_optimizer(10, 5)