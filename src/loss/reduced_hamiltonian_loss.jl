@doc raw"""
    ReducedHamiltonianLoss(hamiltonian; λ = 1, h = 1e-4)

Make an instance of `ReducedHamiltonianLoss` for a Hamiltonian ``H`` on the full space.

The loss fits a [`Decoder`](@ref) ``\Psi^\mathrm{dec}:\mathbb{R}^{2n}\to\mathbb{R}^{2N}`` on latent points ``z`` to target states ``x^*`` and to target gradients ``g^*`` of the *reduced Hamiltonian* ``H\circ\Psi^\mathrm{dec}``:

```math
L = \frac{\lVert\Psi^\mathrm{dec}(z) - x^*\rVert}{\lVert x^*\rVert}
  + \lambda\,\frac{\lVert\nabla_z(H\circ\Psi^\mathrm{dec})(z) - g^*\rVert}{\lVert g^*\rVert},
```

with the absolute error in place of a relative one whose target vanishes on the batch (``g^* = 0`` where every latent point of the batch is a critical point of the target).

For a symplectic decoder, as in a [`SymplecticAutoencoder`](@ref), the reduced vector field is ``\mathbb{J}_{2n}\nabla_z(H\circ\Psi^\mathrm{dec})``, so the second term fits the reduced dynamics directly: their orbits are the level sets of ``H\circ\Psi^\mathrm{dec}``, and a small error in its gradient gives nearly the same phase portrait, its critical points and separatrices included. A fit of the states alone can be small while this gradient is not, in particular where ``H\circ\Psi^\mathrm{dec}`` is steep.

The gradient with respect to ``z`` is computed by central differences with step `h`, from ``2\cdot2n + 1`` evaluations of the decoder on the batch. This keeps the loss differentiable with respect to the parameters by the same reverse-mode pass as any other loss, without nesting automatic differentiation.

# Arguments

- `hamiltonian`: a function that maps a matrix of states in ``\mathbb{R}^{2N}``, one per column, to the vector of their energies. It has to be differentiable by Zygote.
- `λ = 1`: the weight of the gradient term.
- `h = 1e-4`: the step of the central differences, in latent coordinates.

# Functor

```julia
loss(model, ps, input, output)
loss(nn, input, output)
```

`input` holds the latent points ``z``, one per column (``2n`` rows). `output` stacks the targets: the first ``2N`` rows are ``x^*``, the last ``2n`` rows are ``g^*``.

# Examples

A decoder fitted to its own states and reduced gradients has a loss of the size of the difference error:

```jldoctest
using GeometricMachineLearning
using ForwardDiff: gradient
import Random
Random.seed!(123)

H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ X[2, :]   # |p|²/2 + q₂ on ℝ⁴
dec = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
z = randn(2, 20)
x = dec(z)
g = reduce(hcat, [gradient(ζ -> H(reshape(dec(ζ), 4, 1))[1], ζ) for ζ in eachcol(z)])
loss = ReducedHamiltonianLoss(H)
loss(dec, z, vcat(x, g)) < 1e-6

# output

true
```
"""
struct ReducedHamiltonianLoss{HT, T} <: NetworkLoss
    hamiltonian::HT
    λ::T
    h::T
end

function ReducedHamiltonianLoss(hamiltonian; λ::Real = 1, h::Real = 1e-4)
    T = float(promote_type(typeof(λ), typeof(h)))
    ReducedHamiltonianLoss(hamiltonian, T(λ), T(h))
end

# The relative error, and the absolute one where the target vanishes: a batch of latent points that
# are all critical points of the target has g* = 0.
_relative_error(prediction, target) = (n = norm(target); iszero(n) ? norm(prediction) : norm(prediction - target) / n)

# The gradient of `H∘model` with respect to each column of `z`, by central differences.
function _reduced_hamiltonian_gradient(loss::ReducedHamiltonianLoss, model, ps, z::AbstractMatrix)
    n = size(z, 1)
    h = convert(eltype(z), loss.h)
    rows = map(1:n) do i
        # the shift is a column broadcast over the batch; built without mutation for Zygote
        δ = map(j -> j == i ? h : zero(h), 1:n)
        Hp = loss.hamiltonian(model(z .+ δ, ps))
        Hm = loss.hamiltonian(model(z .- δ, ps))
        reshape((Hp .- Hm) ./ (2h), 1, :)
    end
    reduce(vcat, rows)
end

function (loss::ReducedHamiltonianLoss)(model::Union{Chain, AbstractExplicitLayer}, ps,
        input::AbstractMatrix, output::AbstractMatrix)
    n = size(input, 1)
    x_target = output[1:(end - n), :]
    g_target = output[(end - n + 1):end, :]
    state_error = _relative_error(model(input, ps), x_target)
    gradient_error = _relative_error(_reduced_hamiltonian_gradient(loss, model, ps, input), g_target)
    state_error + loss.λ * gradient_error
end

# A `DataLoader` built from an input and an output matrix holds them as tensors with a trailing
# dimension of one; the columns are independent latent points either way.
function (loss::ReducedHamiltonianLoss)(model::Union{Chain, AbstractExplicitLayer}, ps,
        input::AbstractArray{T, 3}, output::AbstractArray{T, 3}) where {T}
    loss(model, ps, reshape(input, size(input, 1), :), reshape(output, size(output, 1), :))
end
