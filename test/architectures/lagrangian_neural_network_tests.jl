using GeometricMachineLearning
using GeometricMachineLearning: _euler_lagrange_acceleration, lagrangian_acceleration,
                                params
using LinearAlgebra
using Test
using Zygote
import Random

Random.seed!(1234)

@doc raw"""
The Euler–Lagrange solve must *be* the Euler–Lagrange equation, not merely run.

For ``L = \tfrac12\dot{q}^TM\dot{q} + q^TC\dot{q} - \tfrac12q^TKq`` the equations
``\tfrac{d}{dt}\partial{}L/\partial\dot{q} = \partial{}L/\partial{}q`` read
``M\ddot{q} + C^T\dot{q} = C\dot{q} - Kq``, so

```math
    \ddot{q} = M^{-1}\left((C - C^T)\dot{q} - Kq\right),
```

in closed form and independently of any network. `C` is deliberately *not* symmetric: with a
symmetric `C` the transpose in the implementation cancels and the test cannot see it.
"""
function test_euler_lagrange_acceleration_against_closed_form(n::Integer)
    M = (A = rand(n, n); A' * A + n * I)
    K = (B = rand(n, n); B' * B + n * I)
    C = rand(n, n)
    q = rand(n)
    q̇ = rand(n)

    H = [-K C; C' M]
    g = vcat(C * q̇ - K * q, M * q̇ + C' * q)
    reference = M \ ((C - C') * q̇ - K * q)

    @test _euler_lagrange_acceleration(H, g, q̇) ≈ reference

    # The same expression with the transpose dropped is a different one, so the assertion above
    # is not passing for a reason unrelated to the index convention.
    without_transpose = M \ (g[1:n] - C * q̇)
    @test !isapprox(without_transpose, reference)
end

for n in (2, 3, 4)
    test_euler_lagrange_acceleration_against_closed_form(n)
end

"""
The compiled symbolic derivatives must agree with `Zygote`'s, per sample and in a batch.
"""
function test_lagrangian_acceleration_against_zygote(n::Integer, batch_size::Integer)
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch)
    acceleration = lagrangian_acceleration(arch)

    input = randn(2n, batch_size)
    predicted = acceleration(input, params(nn))
    @test size(predicted) == (n, batch_size)

    for b in 1:batch_size
        x = input[:, b]
        H = Zygote.hessian(y -> sum(nn(y, params(nn))), x)
        g = Zygote.gradient(y -> sum(nn(y, params(nn))), x)[1]
        @test predicted[:, b] ≈ _euler_lagrange_acceleration(H, g, x[(n + 1):(2n)])
    end
end

test_lagrangian_acceleration_against_zygote(1, 4)
test_lagrangian_acceleration_against_zygote(2, 3)

"""
Trailing input dimensions must be carried through. `Batch` hands the loss a `(2n, s, B)` array, not
a matrix, so the documented reshape is on the path every training run takes.
"""
function test_lagrangian_acceleration_keeps_trailing_dimensions(n::Integer = 2)
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch)
    acceleration = lagrangian_acceleration(arch)

    input = randn(2n, 3, 5)
    predicted = acceleration(input, params(nn))
    @test size(predicted) == (n, 3, 5)
    @test reshape(predicted, n, :) ≈ acceleration(reshape(input, 2n, :), params(nn))
end

test_lagrangian_acceleration_keeps_trailing_dimensions()

@doc raw"""
The solve inverts ``\nabla_{\dot{q}}\nabla_{\dot{q}}L``, so the acceleration is only as trustworthy
as that block's conditioning. This is what decided the solved form over an un-inverted residual, so
it is asserted rather than left in a commit message.
"""
function test_velocity_hessian_is_well_conditioned(n::Integer; draws = 10, points = 10)
    # The architecture matches the tests above, so `Zygote.hessian` is already specialised for it.
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    worst = 0.0
    for _ in 1:draws
        nn = NeuralNetwork(arch)
        for _ in 1:points
            H = Zygote.hessian(y -> sum(nn(y, params(nn))), randn(2n))
            worst = max(worst, cond(H[(n + 1):(2n), (n + 1):(2n)]))
        end
    end
    @test worst < 1.0e8
end

for n in (1, 2)
    test_velocity_hessian_is_well_conditioned(n)
end

"""
`LNNLoss` must be the default loss of the architecture, be callable, and be differentiable with
respect to the parameters.
"""
function test_lnn_loss(n::Integer = 1)
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch)
    loss = LNNLoss(arch)

    @test typeof(loss) <: NetworkLoss
    @test typeof(NetworkLoss(arch)) <: LNNLoss

    input = randn(2n, 8)
    output = randn(n, 8)
    @test typeof(loss(params(nn), input, output)) <: Real
    @test loss(nn.model, params(nn), input, output) == loss(params(nn), input, output)

    dp = Zygote.gradient(ps -> loss(ps, input, output), params(nn))[1]
    @test typeof(dp) <: NetworkParameters
    @test keys(dp) == keys(params(nn))
    @test !all(iszero, dp.L1.W)
end

test_lnn_loss(1)
test_lnn_loss(2)

"""
The loss must reach zero on data a Lagrangian can actually reproduce, so train against the
accelerations the *network's own* Euler–Lagrange equations give at its initial parameters. Those
are reachable by construction, which makes "the loss decreases" a statement about training rather
than about how hard the target was.
"""
function test_lnn_training(; n = 1, n_epochs = 30)
    # The width and depth deliberately match the tests above. A different `Chain` length is a fresh
    # `Zygote` specialisation of the whole loss, which costs 61 s of this file's wall time and buys
    # nothing: the assertion is that training reduces the loss, and that does not depend on either.
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch)
    loss = LNNLoss(arch)

    input = randn(2n, 60)
    output = loss.acceleration(input, params(nn))

    # Start the training from a different draw, so the target is not already met.
    nn = NeuralNetwork(arch)
    dl = DataLoader(input, output; suppress_info = true)
    optimizer = Optimizer(Adam(), nn)
    loss_array = optimizer(nn, dl, Batch(10), n_epochs, loss; show_progress = false)

    @test loss_array[end] < loss_array[1]
end

test_lnn_training()
