using GeometricMachineLearning
using Zygote
using Test
using LinearAlgebra: norm

@doc raw"""
This tests the BFGS optimizer.
"""
function bfgs_optimizer(N)
    B = inv(rand(N, N))
    loss(A) = norm(A - B) ^ (3)
    A = randn(N, N)
    loss1 = loss(A)
    opt = BFGSOptimizer(1e-3)
    optimizer_instance = Optimizer(opt, A)
    ∇loss = gradient(loss, A)[1]
    GeometricMachineLearning.bfgs_initialization!(optimizer_instance, optimizer_instance.cache, ∇loss)
    A .= A + ∇loss
    loss2 = loss(A)
    ∇loss = gradient(loss, A)[1]
    GeometricMachineLearning.bfgs_update!(optimizer_instance, optimizer_instance.cache, ∇loss)
    A .= A + ∇loss
    loss3 = loss(A)
    @test loss1 > loss2 > loss3
end

bfgs_optimizer(10)

function bfgs_optimizer2(N, n_iterates=10)
    losses = zeros(n_iterates+2)
    B = inv(rand(N, N))
    loss(A) = norm(A - B) ^ (3)
    A = randn(N, N)
    losses[1] = loss(A)
    opt = BFGSOptimizer(1e-3)
    optimizer_instance = Optimizer(opt, A)
    ∇loss = gradient(loss, A)[1]
    GeometricMachineLearning.bfgs_initialization!(optimizer_instance, optimizer_instance.cache, ∇loss)
    A .= A + ∇loss
    losses[2] = loss(A)
    for i in 1:n_iterates
        ∇loss = gradient(loss, A)[1]
        GeometricMachineLearning.bfgs_update!(optimizer_instance, optimizer_instance.cache, ∇loss)
        A .= A + ∇loss
        losses[i+2] = loss(A)
    end
    losses
end

# bfgs_optimizer2(10)