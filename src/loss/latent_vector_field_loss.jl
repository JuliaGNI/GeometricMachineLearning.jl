@doc raw"""
    LatentVectorFieldLoss(arch, hamiltonian; λ = 1, h = nothing, vector_field_scale = 1, reconstruction = true, energy = false, reference_state = nothing, energy_scale = 1)

Make an instance of `LatentVectorFieldLoss` for a symplectic autoencoder architecture `arch` and a Hamiltonian ``H`` on the full space.

The loss trains encoder and decoder together on states ``x`` of the full system and on its vector field ``X_H(x) = \mathbb{J}_{2N}\nabla{}H(x)`` there. With ``\xi = \Psi^\mathrm{enc}(x)`` it is

```math
L = \frac{\lVert\Psi^\mathrm{dec}(\xi) - x\rVert}{\lVert x\rVert}
  + \lambda\,\frac{\lVert\mathbb{J}_{2n}\nabla_\xi(H\circ\Psi^\mathrm{dec})(\xi) - \nabla\Psi^\mathrm{enc}(x)X_H(x)\rVert}{\sqrt{B}\,s},
```

where the norms are taken over the batch of ``B`` states and ``s`` is `vector_field_scale`. The first term is the relative reconstruction error. The second term compares two vector fields in the latent space: the *reduced vector field* ``\mathbb{J}_{2n}\nabla_\xi(H\circ\Psi^\mathrm{dec})``, which is the vector field of the reduced system for a symplectic decoder, and the full vector field pushed forward by the encoder. It vanishes when the encoder maps the trajectories of the full system onto those of the reduced one. No target decoder and no latent targets are needed, only states and the full vector field.

The second term is the root-mean-square error of the latent vector field over the batch, divided by a scale fixed when the loss is built. It is continuous in ``X_H``, also where ``X_H`` vanishes, and a minibatch and the whole data set give values of the same size. With ``s = 1`` it is the absolute error. The target ``\nabla\Psi^\mathrm{enc}(x)X_H(x)`` changes with the encoder, so no scale makes the term relative for every encoder; the root-mean-square of ``X_H`` over the data set of ``N_\mathrm{data}`` states, ``s = \lVert X_H\rVert/\sqrt{N_\mathrm{data}}``, is a natural choice for an encoder whose derivative has a norm of order one, such as the transpose of an orthonormal basis.

With `energy = true` the loss has a third term, the energy error

```math
\frac{\lVert(\tilde{H}(\xi) - \tilde{H}(\xi_\mathrm{ref})) - (H(x) - H(x_\mathrm{ref}))\rVert}{\sqrt{B}\,s_E},
```

with the reduced Hamiltonian ``\tilde{H} = H\circ\Psi^\mathrm{dec}``, the reference state ``x_\mathrm{ref}`` = `reference_state`, its encoding ``\xi_\mathrm{ref} = \Psi^\mathrm{enc}(x_\mathrm{ref})``, and ``s_E`` = `energy_scale`. A Hamiltonian is defined up to an additive constant, which the dynamics do not see. The term compares energy differences against one fixed reference point, so it does not depend on that constant either; this is the normalisation of a learned Hamiltonian at a reference point of C. Offen and S. Ober-Blöbaum, "Symplectic integration of learned Hamiltonian systems", Chaos 32(1), 013122 (2022), doi:10.1063/5.0065913, Sec. III, Eq. (12). The reference is fixed when the loss is built, not taken from the batch, so the term of a state does not depend on the other states of its batch. The term asks that the reduced Hamiltonian take the energy of the state it encodes. Unlike the reconstruction it does not tie ``\Psi^\mathrm{dec}(\xi)`` to ``x``, and unlike the vector-field term it keeps its weight where the vector field vanishes.

!!! warning "Reconstruction and dynamics can contradict each other"
    The reconstruction term asks that ``\Psi^\mathrm{dec}(\Psi^\mathrm{enc}(x)) = x``; the vector-field and energy terms ask that the reduced Hamiltonian system reproduce the full system's trajectories and energies. Where no symplectic chart can do both (for instance where the data contain a separatrix that a single chart cannot reconstruct), a network trained on both must fail one of them. The weight `λ` and the scales `vector_field_scale` and `energy_scale` decide which; they also carry the units of the vector field and of the energy, which the reconstruction term does not have.

!!! warning "The vector field has to be the one the data follow"
    When the states lie on a submanifold, as the states of a constrained system do, the target is the system's vector field there, which is tangent to it, and not ``\mathbb{J}_{2N}\nabla{}H`` of a Hamiltonian that merely restricts to the right energy on it. The push-forward of a field that leaves the submanifold involves the derivatives of the encoder off the data, which do not describe how encoded trajectories move.

Both derivatives are central differences with the step `h`. The gradient ``\nabla_\xi(H\circ\Psi^\mathrm{dec})`` takes ``2\cdot2n`` decoder evaluations with the step `h` in latent coordinates. The push-forward is ``\nabla\Psi^\mathrm{enc}(x)X_H(x) \approx (\Psi^\mathrm{enc}(x + hX_H) - \Psi^\mathrm{enc}(x - hX_H))/(2h)``, from two encoder evaluations; its step in the full space is ``h\lVert X_H(x)\rVert``, so it scales with the vector field. This keeps the loss differentiable with respect to the parameters by the same reverse-mode pass as any other loss, without nesting automatic differentiation.

# Arguments

- `arch`: the architecture whose network is trained, a [`SymplecticAutoencoder`](@ref) or a [`PSDArch`](@ref). It fixes where the encoder ends in the chain and the latent dimension ``2n``. Another symplectic architecture raises an `ArgumentError`, and an architecture that is not symplectic has no method.
- `hamiltonian`: a function that maps a matrix of states in ``\mathbb{R}^{2N}``, one per column, to the vector of their energies. It has to be differentiable by Zygote.
- `λ = 1`: the weight of the vector-field term, finite and ``\geq 0``.
- `h = nothing`: the step of the central differences, finite and ``> 0``. With `nothing`, the step is `cbrt(eps(T))` for the element type `T` of the data, which balances the truncation and the round-off error of the central differences.
- `vector_field_scale = 1`: the scale ``s`` of the vector-field term, finite and ``> 0``.
- `reconstruction = true`: with `false` the reconstruction term is left out. Only the reduced dynamics (and, with `energy = true`, the energies) are then fitted; nothing ties ``\Psi^\mathrm{dec}(\xi)`` to ``x`` except through ``H\circ\Psi^\mathrm{dec}``.
- `energy = false`: with `true` the loss has the energy term.
- `reference_state = nothing`: the reference state ``x_\mathrm{ref}`` of the energy term, a vector of the full dimension ``2N`` with finite entries. It is required with `energy = true`.
- `energy_scale = 1`: the scale ``s_E`` of the energy term, finite and ``> 0``. The root-mean-square of ``H(x) - H(x_\mathrm{ref})`` over the data set is a natural choice.

Another `λ`, `h`, `vector_field_scale`, `energy_scale` or `reference_state`, and `energy = true` without a `reference_state`, raise an `ArgumentError`.

# Functor

```julia
loss(model, ps, input, output)
loss(nn, input, output)
```

`input` holds the states ``x``, one per column (``2N`` rows), and `output` the vector field ``X_H(x)`` at them, of the same size; another size raises an `ArgumentError`, and so does a network that does not have the number of layers of `arch` or whose encoder does not give its latent dimension. The model is the whole autoencoder. The loss has the element type of the data, for a network of the same element type. Both forms also take the 3-tensors of a `DataLoader` built from an input and an output matrix; ``B`` then counts the columns of all batches.

# Examples

After `solve!`, the encoder of a proper symplectic decomposition is the transpose ``A^T`` of the symplectic and orthonormal basis ``A`` of the decoder, and ``A^T\mathbb{J}_{2N} = \mathbb{J}_{2n}A^T``. For a state in the range of the decoder, the reconstruction is therefore exact, and the pushed-forward field ``A^T\mathbb{J}_{2N}\nabla{}H`` is the reduced field ``\mathbb{J}_{2n}A^T\nabla{}H``, for every Hamiltonian. The loss vanishes up to the error of the central differences:

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: solve!
import Random
Random.seed!(123)

H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ cos.(X[1, :])   # |p|²/2 + cos(q₁) on ℝ⁴
XH(X) = vcat(X[3:4, :], sin.(X[1:1, :]), zero(X[2:2, :]))           # 𝕁∇H = (p, -∇_q H)
arch = PSDArch(4, 2)
nn = NeuralNetwork(arch)
solve!(nn, randn(4, 20))
x = decoder(nn)(randn(2, 10))
loss = LatentVectorFieldLoss(arch, H)
loss(nn, x, XH(x)) < 1e-8

# output

true
```
"""
struct LatentVectorFieldLoss{
    N, HT, T <: AbstractFloat, ST <: Union{Nothing, AbstractFloat},
    RT <: Union{Nothing, Vector{<:AbstractFloat}}} <: NetworkLoss
    hamiltonian::HT
    λ::T
    h::ST
    vector_field_scale::T
    energy_scale::T
    reconstruction::Bool
    energy::Bool
    n_layers::Int
    reduced_dim::Int
    reference_state::RT

    function LatentVectorFieldLoss(arch::SymplecticCompression, hamiltonian; λ::Real = 1,
            h::Union{Nothing, Real} = nothing, vector_field_scale::Real = 1,
            reconstruction::Bool = true, energy::Bool = false,
            reference_state::Union{Nothing, AbstractVector{<:Real}} = nothing,
            energy_scale::Real = 1)
        (isfinite(λ) && λ >= 0) || throw(ArgumentError(
            "LatentVectorFieldLoss requires a finite weight λ >= 0, got λ = $λ."))
        (h === nothing || (isfinite(h) && h > 0)) || throw(ArgumentError(
            "LatentVectorFieldLoss requires a finite step h > 0, got h = $h."))
        (isfinite(vector_field_scale) && vector_field_scale > 0) || throw(ArgumentError(
            "LatentVectorFieldLoss requires a finite vector_field_scale > 0, got vector_field_scale = $vector_field_scale."))
        (isfinite(energy_scale) && energy_scale > 0) || throw(ArgumentError(
            "LatentVectorFieldLoss requires a finite energy_scale > 0, got energy_scale = $energy_scale."))
        (energy && reference_state === nothing) && throw(ArgumentError(
            "LatentVectorFieldLoss with energy = true requires a reference_state, the full state x_ref at which the energies are compared."))
        if reference_state !== nothing
            length(reference_state) == arch.full_dim || throw(ArgumentError(
                "LatentVectorFieldLoss requires a reference_state of length $(arch.full_dim), the full dimension of the architecture, got length $(length(reference_state))."))
            all(isfinite, reference_state) || throw(ArgumentError(
                "LatentVectorFieldLoss requires a reference_state with finite entries."))
        end
        N = _latent_vf_n_encoder_layers(arch)
        n_layers = N + length(decoder_model(arch).layers)
        T = float(promote_type(typeof(λ), typeof(vector_field_scale), typeof(energy_scale)))
        h_float = h === nothing ? nothing : float(h)
        x_ref = reference_state === nothing ? nothing : float.(collect(reference_state))
        new{N, typeof(hamiltonian), T, typeof(h_float), typeof(x_ref)}(
            hamiltonian, T(λ), h_float, T(vector_field_scale), T(energy_scale),
            reconstruction, energy, n_layers, arch.reduced_dim, x_ref)
    end
end

# The number of layers of the encoder in the chain of the network of `arch`.
function _latent_vf_n_encoder_layers(arch::Union{SymplecticAutoencoder, PSDArch})
    length(encoder_model(arch).layers)
end
function _latent_vf_n_encoder_layers(arch::SymplecticCompression)
    throw(ArgumentError("LatentVectorFieldLoss takes a SymplecticAutoencoder or a PSDArch, got a $(nameof(typeof(arch)))."))
end

# The encoder and the decoder of the chain, each with its parameters.
function _split_autoencoder(loss::LatentVectorFieldLoss{N}, model::Chain, ps) where {N}
    vals = values(ps)
    M = length(model.layers)
    enc = Chain(ntuple(i -> model.layers[i], Val(N))...)
    dec = Chain(ntuple(i -> model.layers[N + i], Val(M - N))...)
    enc, ntuple(i -> vals[i], Val(N)), dec, ntuple(i -> vals[N + i], Val(M - N))
end

# 𝕁_{2n} g = (g_p, -g_q) for g = (g_q, g_p).
function _poisson_latent(g::AbstractMatrix)
    (n = size(g, 1) ÷ 2; vcat(g[(n + 1):end, :], -g[1:n, :]))
end

# The norm over the batch of the energy differences against the reference state, of the reduced
# Hamiltonian at the encoded states and of the Hamiltonian at the states.
function _energy_difference_error(loss::LatentVectorFieldLoss, enc, ps_enc, dec, ps_dec,
        input::AbstractMatrix{T}, decoded) where {T}
    H = loss.hamiltonian
    x_ref = reshape(T.(loss.reference_state), :, 1)
    decoded_ref = dec(enc(x_ref, ps_enc), ps_dec)
    norm((H(decoded) .- H(decoded_ref)) .- (H(input) .- H(x_ref)))
end

function (loss::LatentVectorFieldLoss)(
        model::Chain, ps, input::AbstractMatrix, output::AbstractMatrix)
    size(output) == size(input) || throw(ArgumentError(
        "LatentVectorFieldLoss requires an output of the size of the input, the vector field X_H(x) at each state x, got an output of size $(size(output)) for an input of size $(size(input))."))
    length(model.layers) == loss.n_layers || throw(ArgumentError(
        "LatentVectorFieldLoss requires a network of $(loss.n_layers) layers, the chain of its architecture, got $(length(model.layers)) layers; build the loss from the architecture of the network."))
    T = promote_type(eltype(input), eltype(output))
    enc, ps_enc, dec, ps_dec = _split_autoencoder(loss, model, ps)
    h = _difference_step(loss.h, T)
    ξ = enc(input, ps_enc)
    size(ξ, 1) == loss.reduced_dim || throw(ArgumentError(
        "LatentVectorFieldLoss requires an encoder with $(loss.reduced_dim) output rows, the latent dimension of its architecture, got $(size(ξ, 1)) rows; build the loss from the architecture of the network."))
    decoded = dec(ξ, ps_dec)
    # the root-mean-square errors over the B columns, in the scales fixed by the loss
    rms = sqrt(T(size(input, 2)))
    reconstruction_error = loss.reconstruction ? _compute_loss(decoded, input) : zero(T)
    energy_error = if loss.energy
        E = _energy_difference_error(loss, enc, ps_enc, dec, ps_dec, input, decoded)
        E / (rms * T(loss.energy_scale))
    else
        zero(T)
    end
    pushed_forward = (enc(input .+ h .* output, ps_enc) .-
                      enc(input .- h .* output, ps_enc)) ./ (2h)
    gradient = _reduced_hamiltonian_gradient(loss.hamiltonian, h, dec, ps_dec, ξ)
    vector_field_error = norm(_poisson_latent(gradient) - pushed_forward) /
                         (rms * T(loss.vector_field_scale))
    reconstruction_error + energy_error + T(loss.λ) * vector_field_error
end

# A `DataLoader` built from an input and an output matrix holds them as tensors with a trailing
# dimension of one; the columns are independent states either way.
function (loss::LatentVectorFieldLoss)(
        model::Chain, ps, input::AbstractArray{T, 3}, output::AbstractArray{T, 3}) where {T}
    loss(model, ps, reshape(input, size(input, 1), :), reshape(output, size(output, 1), :))
end
