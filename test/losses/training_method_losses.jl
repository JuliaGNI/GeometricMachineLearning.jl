using GeometricMachineLearning
using GeometricMachineLearning: _symplectic_euler_evaluation_point,
                                _discrete_lagrangian_derivatives,
                                hamiltonian_vector_field, params
using LinearAlgebra
using Test
using Zygote
import Random

Random.seed!(1234)

# ---------------------------------------------------------------- ambiguity with `NetworkLoss`

@doc raw"""
Each three-argument loss functor `(loss)(ps, input, output)` must not collide with
`AbstractNeuralNetworks`' own `(::NetworkLoss)(::NeuralNetwork, input, output)`. With an untyped
first argument the two are ambiguous -- same argument count, neither more specific -- and
`loss(nn, input, output)` is a `MethodError: ... is ambiguous` rather than a call. Typing the
first parameter as `Union{NetworkParameters, NamedTuple}` excludes a `NeuralNetwork`, so the call
resolves to upstream's method, which itself forwards to `loss(nn.params, input, output)` -- the
two must therefore return the same number.
"""
function test_loss_of_a_network_matches_loss_of_its_parameters()
    hnn_arch = StandardHamiltonianArchitecture(2, 4, 1)
    nn = NeuralNetwork(hnn_arch)
    loss = HNNLoss(hnn_arch)
    input, output = randn(2, 5), randn(2, 5)
    @test loss(nn, input, output) == loss(params(nn), input, output)

    lnn_arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
    nn = NeuralNetwork(lnn_arch)
    loss = LNNLoss(lnn_arch)
    input, output = randn(2, 5), randn(1, 5)
    @test loss(nn, input, output) == loss(params(nn), input, output)

    se_arch = StandardHamiltonianArchitecture(2, 4, 1)
    nn = NeuralNetwork(se_arch)
    loss = SymplecticEulerLoss(se_arch, 0.1)
    input, output = randn(2, 5), randn(2, 5)
    @test loss(nn, input, output) == loss(params(nn), input, output)

    vm_arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
    nn = NeuralNetwork(vm_arch)
    loss = VariationalMidpointLoss(vm_arch, 0.05)
    input, output = randn(2, 5), randn(1, 5)
    @test loss(nn, input, output) == loss(params(nn), input, output)
end

test_loss_of_a_network_matches_loss_of_its_parameters()

# ---------------------------------------------------------------- SymplecticEulerLoss

@doc raw"""
The loss must vanish on a trajectory the learned Hamiltonian actually generates.

Take the network's *own* Hamiltonian vector field, step it with symplectic Euler A, and feed the
resulting pair back in: the residual is then zero by construction, so a non-zero value would mean
the loss is not the method it claims to be.

Symplectic Euler A is implicit in ``q``, so the step is taken by fixed-point iteration.
"""
function symplectic_euler_a_step(hvf, ps, state, Δt; iterations = 60)
    n = length(state) ÷ 2
    q, p = state[1:n], state[(n + 1):(2n)]
    q_next = copy(q)
    for _ in 1:iterations
        field = hvf(vcat(q_next, p), ps)
        q_next = q + Δt * field[1:n]
    end
    field = hvf(vcat(q_next, p), ps)
    vcat(q_next, p + Δt * field[(n + 1):(2n)])
end

function test_symplectic_euler_loss_vanishes_on_its_own_step(n::Integer = 1; Δt = 1.0e-3)
    arch = StandardHamiltonianArchitecture(2n, 4, 1)
    nn = NeuralNetwork(arch)
    loss = SymplecticEulerLoss(arch, Δt)
    hvf = hamiltonian_vector_field(arch)

    input = reduce(hcat, [randn(2n) for _ in 1:6])
    output = reduce(hcat,
        [symplectic_euler_a_step(hvf, params(nn), input[:, b], Δt) for b in axes(input, 2)])

    @test loss(params(nn), input, output) < 1.0e-8

    # A trajectory that is *not* the method's own step must not pass, so the assertion above is
    # not vacuous.
    @test loss(params(nn), input, output + 0.1 * randn(size(output))) > 1.0e-3
end

test_symplectic_euler_loss_vanishes_on_its_own_step(1)
test_symplectic_euler_loss_vanishes_on_its_own_step(2)

"""
The two variants must differ, and each must take its half of the state from the right array.
"""
function test_symplectic_euler_variants(n::Integer = 2)
    arch = StandardHamiltonianArchitecture(2n, 4, 1)
    nn = NeuralNetwork(arch)
    input = randn(2n, 5)
    output = randn(2n, 5)

    loss_a = SymplecticEulerLoss(arch, 0.1)
    loss_b = SymplecticEulerLoss(arch, 0.1; variant = :B)

    point_a = _symplectic_euler_evaluation_point(loss_a, input, output)
    point_b = _symplectic_euler_evaluation_point(loss_b, input, output)
    @test point_a == vcat(output[1:n, :], input[(n + 1):(2n), :])
    @test point_b == vcat(input[1:n, :], output[(n + 1):(2n), :])
    @test loss_a(params(nn), input, output) != loss_b(params(nn), input, output)

    @test_throws ArgumentError SymplecticEulerLoss(arch, 0.1; variant = :C)
end

test_symplectic_euler_variants()

"""
Both variants must be differentiable with respect to the parameters. Training is the whole point of
a loss, and a residual built on a symbolic vector field is exactly the kind of expression an AD
system can refuse.
"""
function test_symplectic_euler_loss_gradient(n::Integer = 1)
    arch = StandardHamiltonianArchitecture(2n, 4, 1)
    nn = NeuralNetwork(arch)
    input = randn(2n, 5)
    output = randn(2n, 5)
    for variant in (:A, :B)
        loss = SymplecticEulerLoss(arch, 0.1; variant = variant)
        dp = Zygote.gradient(ps -> loss(ps, input, output), params(nn))[1]
        @test typeof(dp) <: NetworkParameters
        @test keys(dp) == keys(params(nn))
        @test !all(iszero, dp.L1.W)
    end
end

test_symplectic_euler_loss_gradient()

"""
A `Float64` timestep must not widen the loss of a `Float32` network. The timestep is a constant of
the method, not data, so the data's element type is what decides the result's.
"""
function test_timestep_does_not_widen_the_element_type(n::Integer = 1)
    hamiltonian = StandardHamiltonianArchitecture(2n, 4, 1)
    ps = params(NeuralNetwork(hamiltonian, Float32))
    input = randn(Float32, 2n, 5)
    output = randn(Float32, 2n, 5)
    @test SymplecticEulerLoss(hamiltonian, 0.1)(ps, input, output) isa Float32
    @test SymplecticEulerLoss(hamiltonian, 0.1f0)(ps, input, output) isa Float32

    lagrangian = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    psl = params(NeuralNetwork(lagrangian, Float32))
    @test VariationalMidpointLoss(lagrangian, 0.05)(
        psl, randn(Float32, 2n, 5), randn(Float32, n, 5)) isa Float32
end

test_timestep_does_not_widen_the_element_type()

function test_symplectic_euler_training(; n = 1, Δt = 1.0e-2, n_epochs = 20)
    arch = StandardHamiltonianArchitecture(2n, 5, 2)
    nn = NeuralNetwork(arch)
    loss = SymplecticEulerLoss(arch, Δt)
    hvf = hamiltonian_vector_field(arch)

    input = reduce(hcat, [randn(2n) for _ in 1:60])
    output = reduce(hcat,
        [symplectic_euler_a_step(hvf, params(nn), input[:, b], Δt) for b in axes(input, 2)])

    nn = NeuralNetwork(arch)
    dl = DataLoader(input, output; suppress_info = true)
    loss_array = Optimizer(Adam(), nn)(
        nn, dl, Batch(10), n_epochs, loss; show_progress = false)
    @test loss_array[end] < loss_array[1]
end

test_symplectic_euler_training()

# ---------------------------------------------------------------- VariationalMidpointLoss

@doc raw"""
The slot derivatives must be the derivatives of the midpoint discrete Lagrangian.

`_discrete_lagrangian_derivatives` writes the chain rule out by hand, so it has to be checked
against a `Zygote` gradient of ``L_d`` itself — which is the thing it exists in order not to call.
"""
function test_discrete_lagrangian_derivatives(n::Integer; Δt = 0.07)
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch)
    loss = VariationalMidpointLoss(arch, Δt)

    qa = randn(n, 1)
    qb = randn(n, 1)
    D₁, D₂ = _discrete_lagrangian_derivatives(loss, params(nn), qa, qb)

    Ld(a, b) = Δt * sum(nn(vcat((a + b) / 2, (b - a) / Δt), params(nn)))
    @test vec(D₁) ≈ Zygote.gradient(a -> Ld(a, qb), qa)[1]
    @test vec(D₂) ≈ Zygote.gradient(b -> Ld(qa, b), qb)[1]
end

for n in (1, 2, 3)
    test_discrete_lagrangian_derivatives(n)
end

@doc raw"""
The loss must *be* the discrete Euler–Lagrange residual, normalised.

Compare it against the same quantity assembled from `Zygote` gradients of ``L_d`` itself, which is
the call `VariationalMidpointLoss` exists in order not to make. This asserts what the loss computes
without needing a discrete trajectory to be solved for first.
"""
function test_variational_loss_is_the_del_residual(n::Integer; Δt = 0.05, batch_size = 4)
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch)
    loss = VariationalMidpointLoss(arch, Δt)

    qₙ = randn(n, batch_size)
    qₙ₊₁ = randn(n, batch_size)
    qₙ₊₂ = randn(n, batch_size)

    Ld(a, b) = Δt * sum(nn(vcat((a + b) / 2, (b - a) / Δt), params(nn)))
    D₂ = reduce(hcat,
        [Zygote.gradient(b -> Ld(qₙ[:, i], b), qₙ₊₁[:, i])[1] for i in 1:batch_size])
    D₁ = reduce(hcat,
        [Zygote.gradient(a -> Ld(a, qₙ₊₂[:, i]), qₙ₊₁[:, i])[1] for i in 1:batch_size])

    @test loss(params(nn), vcat(qₙ, qₙ₊₁), qₙ₊₂) ≈ norm(D₂ + D₁) / norm(vcat(D₂, D₁))
end

for n in (1, 2, 3)
    test_variational_loss_is_the_del_residual(n)
end

@doc raw"""
The loss must vanish on a trajectory that satisfies the discrete Euler–Lagrange equations.

At one degree of freedom the residual is scalar, so ``q_{n+2}`` is found by bracketing a sign change
and bisecting. Bisection cannot diverge, which Newton on this residual does: a single overshoot puts
the midpoint velocity deep into `tanh` saturation, where the gradient is constant to machine
precision and a numerical Jacobian is exactly zero.
"""
function variational_next_position(loss, ps, qₙ, qₙ₊₁)
    _, D₂ = _discrete_lagrangian_derivatives(loss, ps, qₙ, qₙ₊₁)
    residual(q) = only(first(_discrete_lagrangian_derivatives(
        loss, ps, qₙ₊₁, reshape([q], 1, 1))) + D₂)

    centre = only(qₙ₊₁)
    candidates = range(centre - 5.0, centre + 5.0; length = 1001)
    values = residual.(candidates)
    crossing = findfirst(i -> values[i] * values[i + 1] < 0, 1:(length(values) - 1))
    crossing === nothing && return nothing

    lo, hi = candidates[crossing], candidates[crossing + 1]
    for _ in 1:60
        mid = (lo + hi) / 2
        if residual(lo) * residual(mid) ≤ 0
            hi = mid
        else
            lo = mid
        end
    end
    reshape([(lo + hi) / 2], 1, 1)
end

"""
    a_discrete_trajectory(loss, ps; Δt, attempts)

A triple `(qₙ, qₙ₊₁, qₙ₊₂)` that solves the discrete Euler–Lagrange equations of `loss`, or
`nothing` if no starting pair out of `attempts` admits one.

Not every starting pair does. The discrete Euler–Lagrange equations of an arbitrary neural-network
Lagrangian are a root-finding problem that need not have a solution near the starting pair, so this
searches rather than assuming.
"""
function a_discrete_trajectory(loss, ps; Δt = 0.05, attempts = 20)
    for _ in 1:attempts
        qₙ = randn(1, 1)
        qₙ₊₁ = qₙ + Δt * randn(1, 1)
        qₙ₊₂ = variational_next_position(loss, ps, qₙ, qₙ₊₁)
        qₙ₊₂ === nothing || return (qₙ, qₙ₊₁, qₙ₊₂)
    end
    nothing
end

function test_variational_loss_vanishes_on_a_discrete_trajectory(; Δt = 0.05)
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch)
    loss = VariationalMidpointLoss(arch, Δt)

    trajectory = a_discrete_trajectory(loss, params(nn); Δt = Δt)
    @test trajectory !== nothing
    trajectory === nothing && return
    qₙ, qₙ₊₁, qₙ₊₂ = trajectory

    @test loss(params(nn), vcat(qₙ, qₙ₊₁), qₙ₊₂) < 1.0e-10
    # A position that does not solve the discrete Euler–Lagrange equations must not pass.
    @test loss(params(nn), vcat(qₙ, qₙ₊₁), qₙ₊₂ .+ 0.1) > 1.0e-3
end

test_variational_loss_vanishes_on_a_discrete_trajectory()

function test_variational_loss_gradient(n::Integer = 1)
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch)
    loss = VariationalMidpointLoss(arch, 0.05)
    input = randn(2n, 6)
    output = randn(n, 6)

    @test typeof(loss) <: NetworkLoss
    @test loss(nn.model, params(nn), input, output) == loss(params(nn), input, output)

    dp = Zygote.gradient(ps -> loss(ps, input, output), params(nn))[1]
    @test typeof(dp) <: NetworkParameters
    @test keys(dp) == keys(params(nn))
    @test !all(iszero, dp.L1.W)
end

test_variational_loss_gradient()

@doc raw"""
Training must reduce the loss on a real trajectory.

The data are the harmonic oscillator, ``q(t) = \cos(t)``, sampled at a fixed step. Asking the
network for a Lagrangian whose discrete Euler–Lagrange equations that trajectory satisfies is the
intended use of this loss, and it does not depend on the network's own initial draw the way a
self-generated trajectory would.
"""
function test_variational_training(; Δt = 0.05, samples = 60, n_epochs = 20)
    arch = LagrangianNeuralNetwork(2; width = 5, nhidden = 2)
    nn = NeuralNetwork(arch)
    loss = VariationalMidpointLoss(arch, Δt)

    t = range(0.0, Δt * (samples + 1); step = Δt)
    q = cos.(t)
    qₙ = reshape(q[1:samples], 1, samples)
    qₙ₊₁ = reshape(q[2:(samples + 1)], 1, samples)
    qₙ₊₂ = reshape(q[3:(samples + 2)], 1, samples)

    dl = DataLoader(vcat(qₙ, qₙ₊₁), qₙ₊₂; suppress_info = true)
    loss_array = Optimizer(Adam(), nn)(
        nn, dl, Batch(10), n_epochs, loss; show_progress = false)
    @test loss_array[end] < loss_array[1]
end

test_variational_training()
