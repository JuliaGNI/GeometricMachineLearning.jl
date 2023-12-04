using GeometricMachineLearning
using Zygote
using Test
using LinearAlgebra: norm, svd

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

function stiefel_optimizer_test(N, n; T=Float32, n_iterates=10, η=1f-2)
    opt=BFGSOptimizer(T(η))
    A = rand(StiefelManifold{T}, N, n)
    B = T(10)*randn(T, N, N)
    ideal_A = svd(B).U[:, 1:n]
    loss(A) = norm(B - A * A' * B)
    ideal_error = norm(ideal_A)
    losses = zeros(n_iterates+2)
    losses[1] = loss(A)
    optimizer_instance = Optimizer(opt, A)
    λA = GlobalSection(A)
    grad_loss = global_rep(λA, rgrad(A, gradient(loss, A)[1]))
    GeometricMachineLearning.bfgs_initialization!(optimizer_instance, optimizer_instance.cache, grad_loss)
    # geodesic for `grad_loss`
    exp_grad_loss = GeometricMachineLearning.geodesic(grad_loss)
    GeometricMachineLearning.apply_section!(A, λA, exp_grad_loss)
    losses[2] = loss(A)
    for i in 1:n_iterates
        λA = GlobalSection(A)
        grad_loss = global_rep(λA, rgrad(A, gradient(loss, A)[1]))
        GeometricMachineLearning.bfgs_update!(optimizer_instance, optimizer_instance.cache, grad_loss)
        # geodesic for `grad_loss`
        exp_grad_loss = GeometricMachineLearning.geodesic(grad_loss)
        GeometricMachineLearning.apply_section!(A, λA, exp_grad_loss)
        losses[i+2] = loss(A)
    end
    losses, ideal_error, check(A)
end

function stiefel_adam_test(N, n; T=Float32, n_iterates=10)
    opt=AdamOptimizer()
    A = rand(StiefelManifold{T}, N, n)
    B = T(10)*randn(T, N, N)
    ideal_A = svd(B).U[:, 1:n]
    loss(A) = norm(B - A * A' * B)
    ideal_error = norm(ideal_A)
    losses = zeros(n_iterates+2)
    losses[1] = loss(A)
    optimizer_instance = Optimizer(opt, A)
    λA = GlobalSection(A)
    grad_loss = global_rep(λA, rgrad(A, gradient(loss, A)[1]))
    GeometricMachineLearning.update!(optimizer_instance, optimizer_instance.cache, grad_loss)
    # geodesic for `grad_loss`
    exp_grad_loss = GeometricMachineLearning.geodesic(grad_loss)
    GeometricMachineLearning.apply_section!(A, λA, exp_grad_loss)
    losses[2] = loss(A)
    for i in 1:n_iterates
        λA = GlobalSection(A)
        grad_loss = global_rep(λA, rgrad(A, gradient(loss, A)[1]))
        GeometricMachineLearning.update!(optimizer_instance, optimizer_instance.cache, grad_loss)
        # geodesic for `grad_loss`
        exp_grad_loss = GeometricMachineLearning.geodesic(grad_loss)
        GeometricMachineLearning.apply_section!(A, λA, exp_grad_loss)
        losses[i+2] = loss(A)
    end
    losses, ideal_error, check(A)
end