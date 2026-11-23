@doc raw"""
    ReducedHamiltonianLoss(hamiltonian; λ = 1, h = nothing, gradient_scale = 1)

Make an instance of `ReducedHamiltonianLoss` for a Hamiltonian ``H`` on the full space.

The loss fits a [`Decoder`](@ref) ``\Psi^\mathrm{dec}:\mathbb{R}^{2n}\to\mathbb{R}^{2N}`` on latent points ``z`` to target states ``x^*`` and to target gradients ``g^*`` of the *reduced Hamiltonian* ``H\circ\Psi^\mathrm{dec}``:

```math
L = \frac{\lVert\Psi^\mathrm{dec}(z) - x^*\rVert}{\lVert x^*\rVert}
  + \lambda\,\frac{\lVert\nabla_z(H\circ\Psi^\mathrm{dec})(z) - g^*\rVert}{\sqrt{B}\,s},
```

where the norms are taken over the batch of ``B`` latent points and ``s`` is `gradient_scale`. The gradient term is the root-mean-square error of the gradient over the batch, divided by a scale fixed when the loss is built. It is continuous in ``g^*``, also at ``g^* = 0``, and a minibatch and the whole data set give values of the same size. With ``s = 1`` it is the absolute error. A natural scale is the root-mean-square of ``g^*`` over the whole data set of ``N_\mathrm{data}`` points, ``s = \lVert g^*\rVert/\sqrt{N_\mathrm{data}}``, which makes the term a relative error over the data set.

For a symplectic decoder, as in a [`SymplecticAutoencoder`](@ref), the reduced vector field is ``\mathbb{J}_{2n}\nabla_z(H\circ\Psi^\mathrm{dec})``, so the second term fits the reduced vector field at the sampled points. A fit of the states alone can be small while this gradient is not, in particular where ``H\circ\Psi^\mathrm{dec}`` is steep.

The gradient with respect to ``z`` is computed by central differences with step `h`, from ``2\cdot2n + 1`` evaluations of the decoder on the batch. This keeps the loss differentiable with respect to the parameters by the same reverse-mode pass as any other loss, without nesting automatic differentiation.

# Arguments

- `hamiltonian`: a function that maps a matrix of states in ``\mathbb{R}^{2N}``, one per column, to the vector of their energies. It has to be differentiable by Zygote.
- `λ = 1`: the weight of the gradient term, finite and ``\geq 0``.
- `h = nothing`: the step of the central differences, in latent coordinates, finite and ``> 0``. With `nothing`, the step is `cbrt(eps(T))` for the element type `T` of the data, which balances the truncation and the round-off error of the central differences.
- `gradient_scale = 1`: the scale ``s`` of the gradient term, finite and ``> 0``.

Another `λ`, `h` or `gradient_scale` raises an `ArgumentError`.

# Functor

```julia
loss(model, ps, input, output)
loss(nn, input, output)
```

`input` holds the latent points ``z``, one per column (``2n`` rows). `output` stacks the targets: the first ``2N`` rows are ``x^*``, the last ``2n`` rows are ``g^*``; another number of rows raises an `ArgumentError`. The loss has the element type of the data, for a network of the same element type. Both forms also take the 3-tensors of a `DataLoader` built from an input and an output matrix; ``B`` then counts the columns of all batches.

# Examples

A decoder fitted to its own states and reduced gradients has a loss of the size of the difference error:

```jldoctest
using GeometricMachineLearning
using Zygote: gradient
import Random
Random.seed!(123)

H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ X[2, :]   # |p|²/2 + q₂ on ℝ⁴
dec = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
z = randn(2, 20)
x = dec(z)
g = reduce(hcat, [gradient(ζ -> H(reshape(dec(ζ), 4, 1))[1], collect(ζ))[1] for ζ in eachcol(z)])
loss = ReducedHamiltonianLoss(H)
loss(dec, z, vcat(x, g)) < 1e-6

# output

true
```
"""
struct ReducedHamiltonianLoss{
    HT, T <: AbstractFloat, ST <: Union{Nothing, AbstractFloat}, GT <: AbstractFloat} <:
       NetworkLoss
    hamiltonian::HT
    λ::T
    h::ST
    gradient_scale::GT

    function ReducedHamiltonianLoss(
            hamiltonian; λ::Real = 1, h::Union{Nothing, Real} = nothing,
            gradient_scale::Real = 1)
        (isfinite(λ) && λ >= 0) || throw(ArgumentError(
            "ReducedHamiltonianLoss requires a finite weight λ >= 0, got λ = $λ."))
        (h === nothing || (isfinite(h) && h > 0)) || throw(ArgumentError(
            "ReducedHamiltonianLoss requires a finite step h > 0, got h = $h."))
        (isfinite(gradient_scale) && gradient_scale > 0) || throw(ArgumentError(
            "ReducedHamiltonianLoss requires a finite gradient_scale > 0, got gradient_scale = $gradient_scale."))
        h_float = h === nothing ? nothing : float(h)
        new{typeof(hamiltonian), typeof(float(λ)),
            typeof(h_float), typeof(float(gradient_scale))}(
            hamiltonian, float(λ), h_float, float(gradient_scale))
    end
end

# The step of the central differences in the element type `T` of the data: the given one, or
# `cbrt(eps(T))`, which balances the truncation error O(h²) against the round-off O(eps(T)/h).
_difference_step(::Nothing, ::Type{T}) where {T} = cbrt(eps(T))
_difference_step(h::Real, ::Type{T}) where {T} = T(h)

# The gradient of `H∘model` with respect to each column of `z`, by central differences.
function _reduced_hamiltonian_gradient(loss::ReducedHamiltonianLoss, model, ps, z::AbstractMatrix)
    n = size(z, 1)
    h = _difference_step(loss.h, eltype(z))
    mapreduce(vcat, 1:n) do i
        # the shift is a column broadcast over the batch; built without mutation for Zygote
        δ = map(j -> j == i ? h : zero(h), 1:n)
        Hp = loss.hamiltonian(model(z .+ δ, ps))
        Hm = loss.hamiltonian(model(z .- δ, ps))
        reshape((Hp .- Hm) ./ (2h), 1, :)
    end
end

function (loss::ReducedHamiltonianLoss)(model::Union{Chain, AbstractExplicitLayer}, ps,
        input::AbstractMatrix, output::AbstractMatrix)
    T = promote_type(eltype(input), eltype(output))
    n = size(input, 1)
    x_prediction = model(input, ps)
    N = size(x_prediction, 1)
    size(output, 1) == N + n || throw(ArgumentError(
        "ReducedHamiltonianLoss requires an output of $(N + n) rows, the $N rows of the target states x* over the $n rows of the target gradients g*, got $(size(output, 1)) rows."))
    x_target = output[1:N, :]
    g_target = output[(N + 1):end, :]
    state_error = _compute_loss(x_prediction, x_target)
    # the root-mean-square error over the B columns, in the scale fixed by the loss
    g_prediction = _reduced_hamiltonian_gradient(loss, model, ps, input)
    B = size(input, 2)
    gradient_error = norm(g_prediction - g_target) / (sqrt(T(B)) * T(loss.gradient_scale))
    state_error + T(loss.λ) * gradient_error
end

# A `DataLoader` built from an input and an output matrix holds them as tensors with a trailing
# dimension of one; the columns are independent latent points either way.
function (loss::ReducedHamiltonianLoss)(model::Union{Chain, AbstractExplicitLayer}, ps,
        input::AbstractArray{T, 3}, output::AbstractArray{T, 3}) where {T}
    loss(model, ps, reshape(input, size(input, 1), :), reshape(output, size(output, 1), :))
end
