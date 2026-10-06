@doc raw"""
    StandardAutoencoder(full_dim, reduced_dim)

Make an instance of `StandardAutoencoder` for dimensions `full_dim` and `reduced_dim`.

# The architecture

A standard autoencoder [goodfellow2016deep](@cite) has an encoder ``\Psi^\mathrm{enc}:\mathbb{R}^N\to\mathbb{R}^n`` and a decoder ``\Psi^\mathrm{dec}:\mathbb{R}^n\to\mathbb{R}^N`` that are both feedforward networks of [`Dense`](@ref) layers, with no structure imposed on either. The encoder is

```math
\Psi^\mathrm{enc} = L^\mathrm{out}_{w\to{}n}\circ{}L_{w\to{}w}\circ\cdots\circ{}L_{w\to{}w}\circ{}L_{N\to{}w},
```

with `n_encoder_layers` nonlinear layers ``L``, the first of which maps ``\mathbb{R}^N`` to the hidden width ``w`` and the others ``\mathbb{R}^w`` to itself, followed by an affine output layer ``L^\mathrm{out}`` without activation. The decoder is built the same way from ``\mathbb{R}^n`` to ``\mathbb{R}^N`` with `n_decoder_layers` nonlinear layers.

Unlike for a [`SymplecticAutoencoder`](@ref), ``\nabla\Psi^\mathrm{dec}`` is in general not symplectic, so the reduced system is not Hamiltonian, and neither ``N`` nor ``n`` has to be even. It is the unstructured counterpart of the [`SymplecticAutoencoder`](@ref), and serves to compare the two.

# Arguments

Besides the required arguments `full_dim` and `reduced_dim` you can provide the following keyword arguments:
- `width::Integer = 2full_dim`: the hidden width ``w``.
- `n_encoder_layers::Integer = 2`: the number of nonlinear layers in the encoder.
- `n_decoder_layers::Integer = 2`: the number of nonlinear layers in the decoder.
- `activation = tanh`: the activation of the nonlinear layers.

# Examples

The two parts compose to the whole network, and the number of parameters follows from the widths:

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: params

arch = StandardAutoencoder(4, 2; width = 10, n_encoder_layers = 2, n_decoder_layers = 3)
nn = NeuralNetwork(arch)
x = rand(4)

nn(x) ≈ decoder(nn)(encoder(nn)(x)), parameterlength(nn.model)

# output

(true, 476)
```

Here the encoder has ``(4 + 1)\cdot10 + (10 + 1)\cdot10 + (10 + 1)\cdot2 = 182`` parameters and the decoder ``(2 + 1)\cdot10 + 2\cdot(10 + 1)\cdot10 + (10 + 1)\cdot4 = 294``.
"""
struct StandardAutoencoder{AT} <: AutoEncoder
    full_dim::Int
    reduced_dim::Int
    width::Int
    n_encoder_layers::Int
    n_decoder_layers::Int
    n_encoder_blocks::Int
    n_decoder_blocks::Int
    activation::AT
end

function StandardAutoencoder(full_dim::Integer, reduced_dim::Integer;
        width::Integer = 2full_dim,
        n_encoder_layers::Integer = 2,
        n_decoder_layers::Integer = 2,
        activation = tanh)
    @assert full_dim ≥ reduced_dim "The dimension of the full-order model has to be at least that of the reduced-order model!"
    @assert n_encoder_layers ≥ 1 && n_decoder_layers ≥ 1 "Encoder and decoder need at least one nonlinear layer each!"
    # The dimension changes once in each direction, from `full_dim` to `reduced_dim`; the hidden
    # layers are not blocks in the sense of `compute_iterations`.
    StandardAutoencoder{typeof(activation)}(full_dim, reduced_dim, width, n_encoder_layers,
        n_decoder_layers, 2, 2, activation)
end

function _standard_autoencoder_layers(arch::StandardAutoencoder, input_dim::Integer,
        output_dim::Integer, n_layers::Integer)
    (Dense(input_dim, arch.width, arch.activation),
        Tuple(Dense(arch.width, arch.width, arch.activation) for _ in 2:n_layers)...,
        Dense(arch.width, output_dim, identity))
end

function encoder_layers_from_iteration(arch::StandardAutoencoder, ::AbstractVector{<:Integer})
    _standard_autoencoder_layers(arch, arch.full_dim, arch.reduced_dim, arch.n_encoder_layers)
end

function decoder_layers_from_iteration(arch::StandardAutoencoder, ::AbstractVector{<:Integer})
    _standard_autoencoder_layers(arch, arch.reduced_dim, arch.full_dim, arch.n_decoder_layers)
end
