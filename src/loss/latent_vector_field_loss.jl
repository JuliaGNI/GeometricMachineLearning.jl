@doc raw"""
    LatentVectorFieldLoss(arch, hamiltonian; λ = 1, h = 1e-4)

Make an instance of `LatentVectorFieldLoss` for an [`AutoEncoder`](@ref) architecture `arch` and a Hamiltonian ``H`` on the full space.

The loss trains encoder and decoder together on states ``x`` of the full system and on its vector field ``X_H(x) = \mathbb{J}_{2N}\nabla{}H(x)`` there. With ``\xi = \Psi^\mathrm{enc}(x)`` it is

```math
L = \frac{\lVert\Psi^\mathrm{dec}(\xi) - x\rVert}{\lVert x\rVert}
  + \lambda\,\frac{\lVert\mathbb{J}_{2n}\nabla_\xi(H\circ\Psi^\mathrm{dec})(\xi) - \nabla\Psi^\mathrm{enc}(x)X_H(x)\rVert}{\lVert\nabla\Psi^\mathrm{enc}(x)X_H(x)\rVert},
```

the reconstruction error plus ``\lambda`` times the error of the *reduced vector field* against the full vector field pushed forward by the encoder, both relative and with the norms taken over the batch.

!!! warning "The vector field has to be the one the data follow"
    When the states lie on a submanifold, as the states of a constrained system do, the target is the system's vector field there, which is tangent to it, and not ``\mathbb{J}_{2N}\nabla{}H`` of a Hamiltonian that merely restricts to the right energy on it. The push-forward of a field that leaves the submanifold involves the derivatives of the encoder off the data, which do not describe how encoded trajectories move. The second term compares the two vector fields in the latent space: it vanishes when the encoder maps the trajectories of the full system onto those of the reduced one, which for a symplectic decoder are the level sets of ``H\circ\Psi^\mathrm{dec}``. No target decoder and no latent targets are needed, only states and the full vector field. Where the pushed-forward field vanishes on the whole batch, the absolute error replaces the relative one.

Both derivatives are central differences with step `h`: ``\nabla_\xi(H\circ\Psi^\mathrm{dec})`` from ``2\cdot2n`` decoder evaluations, and ``\nabla\Psi^\mathrm{enc}(x)X_H(x) \approx (\Psi^\mathrm{enc}(x + hX_H) - \Psi^\mathrm{enc}(x - hX_H))/(2h)`` from two encoder evaluations. This keeps the loss differentiable with respect to the parameters by the same reverse-mode pass as any other loss, without nesting automatic differentiation.

# Arguments

- `arch`: the [`AutoEncoder`](@ref) whose network is trained; it fixes where the encoder ends in the chain.
- `hamiltonian`: a function that maps a matrix of states in ``\mathbb{R}^{2N}``, one per column, to the vector of their energies. It has to be differentiable by Zygote.
- `λ = 1`: the weight of the vector-field term.
- `h = 1e-4`: the step of the central differences.

# Functor

```julia
loss(model, ps, input, output)
loss(nn, input, output)
```

`input` holds the states ``x``, one per column (``2N`` rows), and `output` the vector field ``X_H(x)`` at them (``2N`` rows). The model is the whole autoencoder.

# Examples

The loss is zero when the vector-field targets are what the network itself predicts, here for the linear Hamiltonian ``H(x) = c\cdot x`` with a constant vector field:

```jldoctest
using GeometricMachineLearning
import Random
Random.seed!(123)

arch = SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)
nn = NeuralNetwork(arch)
loss = LatentVectorFieldLoss(arch, X -> vec(sum(X; dims = 1)))
x = randn(4, 10)
v = ones(4, 10)
loss(nn, x, v) > 0

# output

true
```
"""
struct LatentVectorFieldLoss{N, HT, T} <: NetworkLoss
    hamiltonian::HT
    λ::T
    h::T
    function LatentVectorFieldLoss{N}(hamiltonian::HT, λ::T, h::T) where {N, HT, T}
        new{N, HT, T}(hamiltonian, λ, h)
    end
end

function LatentVectorFieldLoss(arch::AutoEncoder, hamiltonian; λ::Real = 1, h::Real = 1e-4)
    T = float(promote_type(typeof(λ), typeof(h)))
    N = length(encoder_model(arch).layers)
    LatentVectorFieldLoss{N}(hamiltonian, T(λ), T(h))
end

n_encoder_layers(::LatentVectorFieldLoss{N}) where {N} = N

# The relative error, and the absolute one where the target vanishes.
_relative_vf_error(prediction, target) = (n = norm(target); iszero(n) ? norm(prediction) : norm(prediction - target) / n)

# The encoder and the decoder of the chain, each with its parameters.
function _split_autoencoder(loss::LatentVectorFieldLoss{N}, model::Chain, ps) where {N}
    vals = values(ps)
    M = length(model.layers)
    enc = Chain(ntuple(i -> model.layers[i], Val(N))...)
    dec = Chain(ntuple(i -> model.layers[N + i], Val(M - N))...)
    enc, ntuple(i -> vals[i], Val(N)), dec, ntuple(i -> vals[N + i], Val(M - N))
end

# ∇_ξ(H∘Ψ^dec) for each column of ξ, by central differences.
function _latent_hamiltonian_gradient(loss::LatentVectorFieldLoss, dec, ps_dec, ξ::AbstractMatrix)
    n = size(ξ, 1)
    h = convert(eltype(ξ), loss.h)
    rows = map(1:n) do i
        # the shift is a column broadcast over the batch; built without mutation for Zygote
        δ = map(j -> j == i ? h : zero(h), 1:n)
        Hp = loss.hamiltonian(dec(ξ .+ δ, ps_dec))
        Hm = loss.hamiltonian(dec(ξ .- δ, ps_dec))
        reshape((Hp .- Hm) ./ (2h), 1, :)
    end
    reduce(vcat, rows)
end

# 𝕁_{2n} g = (g_p, -g_q) for g = (g_q, g_p).
_poisson_latent(g::AbstractMatrix) = (n = size(g, 1) ÷ 2; vcat(g[(n + 1):end, :], -g[1:n, :]))

function (loss::LatentVectorFieldLoss)(model::Chain, ps, input::AbstractMatrix, output::AbstractMatrix)
    enc, ps_enc, dec, ps_dec = _split_autoencoder(loss, model, ps)
    h = convert(eltype(input), loss.h)
    ξ = enc(input, ps_enc)
    reconstruction_error = _relative_vf_error(dec(ξ, ps_dec), input)
    pushed_forward = (enc(input .+ h .* output, ps_enc) .- enc(input .- h .* output, ps_enc)) ./ (2h)
    reduced = _poisson_latent(_latent_hamiltonian_gradient(loss, dec, ps_dec, ξ))
    reconstruction_error + loss.λ * _relative_vf_error(reduced, pushed_forward)
end

# A `DataLoader` built from an input and an output matrix holds them as tensors with a trailing
# dimension of one; the columns are independent states either way.
function (loss::LatentVectorFieldLoss)(model::Chain, ps, input::AbstractArray{T, 3}, output::AbstractArray{T, 3}) where {T}
    loss(model, ps, reshape(input, size(input, 1), :), reshape(output, size(output, 1), :))
end
