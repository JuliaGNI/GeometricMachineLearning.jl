@doc raw"""
    SymplecticEulerLoss <: NetworkLoss

The loss that trains a Hamiltonian neural network on a *trajectory*, through one step of a
symplectic Euler method.

[`HNNLoss`](@ref) needs ``(\dot{q}, \dot{p})`` in the data. This one does not: it needs two
consecutive states and the timestep between them, and it asks that one symplectic Euler step of the
learned Hamiltonian carry the first state to the second.

Variant `:A` evaluates the Hamiltonian vector field at ``(q_{n+1}, p_n)`` and variant `:B` at
``(q_n, p_{n+1})``. Writing ``X_H`` for that vector field and ``\Delta{}t`` for the timestep, the
residual of variant `:A` is

```math
    X_H(q_{n+1}, p_n) - \frac{1}{\Delta{}t}
        \begin{pmatrix} q_{n+1} - q_n \\ p_{n+1} - p_n \end{pmatrix},
```

and the loss is its norm relative to the norm of the finite difference, so it is scale invariant in
the same way [`HNNLoss`](@ref) is.

# Constructor

```julia
SymplecticEulerLoss(arch, timestep)                 # variant :A
SymplecticEulerLoss(arch, timestep; variant = :B)
```

where `arch` is a [`HamiltonianArchitecture`](@ref).

# Functor

```julia
loss(model, ps, input, output)
loss(ps, input, output) # equivalent to the above
```

`input` is ``(q_n, p_n)`` stacked and `output` is ``(q_{n+1}, p_{n+1})`` stacked, so both have
``2n`` rows.
"""
struct SymplecticEulerLoss{V, FT, T} <: NetworkLoss
    hvf::FT
    timestep::T

    function SymplecticEulerLoss(arch::HamiltonianArchitecture, timestep::Real;
            variant::Symbol = :A)
        variant in (:A, :B) ||
            throw(ArgumentError("variant must be :A or :B, got :$(variant)"))
        hvf = hamiltonian_vector_field(arch)
        new{variant, typeof(hvf), typeof(timestep)}(hvf, timestep)
    end
end

@doc raw"""
    _symplectic_euler_evaluation_point(loss, input, output)

The staggered state at which the Hamiltonian vector field is evaluated: ``(q_{n+1}, p_n)`` for
variant `:A`, ``(q_n, p_{n+1})`` for variant `:B`.

Taking one half of the state from `input` and the other from `output` is what makes the step
implicit, and is the whole difference from an explicit Euler residual.
"""
function _symplectic_euler_evaluation_point(
        ::SymplecticEulerLoss{:A}, input::AbstractArray, output::AbstractArray)
    n = size(input, 1) ÷ 2
    vcat(selectdim(output, 1, 1:n), selectdim(input, 1, (n + 1):(2n)))
end

function _symplectic_euler_evaluation_point(
        ::SymplecticEulerLoss{:B}, input::AbstractArray, output::AbstractArray)
    n = size(input, 1) ÷ 2
    vcat(selectdim(input, 1, 1:n), selectdim(output, 1, (n + 1):(2n)))
end

function (loss::SymplecticEulerLoss)(::Union{Chain, AbstractExplicitLayer},
        ps, input::AbstractArray, output::AbstractArray)
    loss(ps, input, output)
end

function (loss::SymplecticEulerLoss)(
        ps::Union{NetworkParameters, NamedTuple}, input::AbstractArray, output::AbstractArray)
    evaluation_point = _symplectic_euler_evaluation_point(loss, input, output)
    # The timestep is converted to the data's element type rather than promoted against it: a
    # `Float64` timestep must not widen the loss of a `Float32` network. `HNNLoss` has no constant
    # to convert, and `symbolic_hamiltonian_vector_field` builds its Poisson tensor from `Int` for
    # this same reason.
    difference_quotient = (output - input) / eltype(input)(loss.timestep)
    field = reshape(
        loss.hvf(reshape(evaluation_point, size(input, 1), :), ps), size(input))
    norm(field - difference_quotient) / norm(difference_quotient)
end
