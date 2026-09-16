@doc raw"""
    VariationalMidpointLoss <: NetworkLoss

The loss that trains a Lagrangian neural network on positions alone, through the discrete
Euler–Lagrange equations of the midpoint discrete Lagrangian.

It needs three consecutive positions and the timestep between them — no velocities and no
accelerations.

The midpoint discrete Lagrangian is

```math
    L_d(q_n, q_{n+1}) = \Delta{}t\,
        L\!\left( \frac{q_n + q_{n+1}}{2}, \frac{q_{n+1} - q_n}{\Delta{}t} \right),
```

and the discrete Euler–Lagrange equations ask that

```math
    D_2L_d(q_n, q_{n+1}) + D_1L_d(q_{n+1}, q_{n+2}) = 0 .
```

The loss is the norm of that residual, relative to the norm of the two derivatives it is built from,
so a Lagrangian scaled by a constant gives the same number.

# Implementation

The two slot derivatives are written out with the chain rule rather than taken with a nested
`Zygote.gradient`:

```math
    D_1L_d = \frac{\Delta{}t}{2}\nabla_qL - \nabla_{\dot{q}}L, \qquad
    D_2L_d = \frac{\Delta{}t}{2}\nabla_qL + \nabla_{\dot{q}}L,
```

both evaluated at the midpoint. Only the *first* derivative of the network with respect to its input
is then needed, and that comes from a compiled symbolic expression exactly as
[`hamiltonian_vector_field`](@ref)'s does. Written the other way — a `Zygote.gradient` inside the
loss — the parameter gradient fails with
`MethodError: no method matching getindex(::IdDict{Any, Any})`.

# Constructor

```julia
VariationalMidpointLoss(arch, timestep)
```

where `arch` is a `LagrangianNeuralNetwork`.

# Functor

```julia
loss(model, ps, input, output)
loss(ps, input, output) # equivalent to the above
```

`input` stacks ``q_n`` on ``q_{n+1}`` and so has ``2n`` rows; `output` is ``q_{n+2}`` and has ``n``.
"""
struct VariationalMidpointLoss{FT, T} <: NetworkLoss
    gradient::FT
    timestep::T

    function VariationalMidpointLoss(arch::LagrangianNeuralNetwork, timestep::Real)
        snn = SymbolicNeuralNetwork(arch)
        ∇L = derivative(SymbolicNeuralNetworks.Jacobian(snn))
        grad = SymbolicNeuralNetworks.build_nn_function(
            ∇L, snn.params, snn.input; inplace = false)
        new{typeof(grad), typeof(timestep)}(grad, timestep)
    end
end

@doc raw"""
    _discrete_lagrangian_derivatives(loss, ps, qa, qb)

Return `(D₁Lᵈ, D₂Lᵈ)` of the midpoint discrete Lagrangian on the pair `(qa, qb)`, one column per
sample.
"""
function _discrete_lagrangian_derivatives(loss::VariationalMidpointLoss, ps,
        qa::AbstractMatrix, qb::AbstractMatrix)
    n, batch_size = size(qa)
    # The timestep is converted to the data's element type rather than promoted against it: a
    # `Float64` timestep must not widen the loss of a `Float32` network.
    Δt = eltype(qa)(loss.timestep)
    midpoint = vcat((qa + qb) / 2, (qb - qa) / Δt)
    g = reshape(loss.gradient(midpoint, ps), 2n, batch_size)
    ∇qL = g[1:n, :]
    ∇q̇L = g[(n + 1):(2n), :]
    (Δt / 2) * ∇qL - ∇q̇L, (Δt / 2) * ∇qL + ∇q̇L
end

function (loss::VariationalMidpointLoss)(::Union{Chain, AbstractExplicitLayer},
        ps, input::AbstractArray, output::AbstractArray)
    loss(ps, input, output)
end

function (loss::VariationalMidpointLoss)(ps, input::AbstractArray, output::AbstractArray)
    n = size(input, 1) ÷ 2
    qₙ = reshape(selectdim(input, 1, 1:n), n, :)
    qₙ₊₁ = reshape(selectdim(input, 1, (n + 1):(2n)), n, :)
    qₙ₊₂ = reshape(output, n, :)

    _, D₂ = _discrete_lagrangian_derivatives(loss, ps, qₙ, qₙ₊₁)
    D₁, _ = _discrete_lagrangian_derivatives(loss, ps, qₙ₊₁, qₙ₊₂)

    norm(D₂ + D₁) / norm(vcat(D₂, D₁))
end
