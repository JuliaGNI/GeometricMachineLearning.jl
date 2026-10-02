@doc raw"""
    PODArch <: AutoEncoder

`PODArch` is the architecture for proper orthogonal decomposition (POD).

## The architecture

POD can be seen as an [`AutoEncoder`](@ref) whose encoder and decoder each consist of a single [`StiefelLayer`](@ref): the decoder is ``x \mapsto Vx`` and the encoder is ``x \mapsto V^Tx`` for some ``V \in St(n, N)``.

## Training

No neural network training is needed: the optimal ``V`` consists of the leading ``n`` left singular vectors of the snapshot matrix (see the docs for [`solve!`](@ref)).

## The constructor

The constructor only takes two arguments as input:
- `full_dim::Integer`
- `reduced_dim::Integer`

# Examples

For two uncoupled harmonic oscillators ``H = (p_1^2 + p_2^2)/2 + (\phi_1q_1^2 + \phi_2q_2^2)/2`` with ``\phi_1 = 0.05``, ``\phi_2 = \pi`` and initial condition ``(q_1, q_2, p_1, p_2) = (0, 0, 1, 3)``, the coordinates with the largest amplitudes are ``q_1`` (amplitude ``1/\sqrt{0.05} = 2\sqrt{5}``) and ``p_2`` (amplitude ``3``). The POD basis is therefore ``[e_1, e_4]`` up to sign:

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: params

ϕ₁, ϕ₂ = 0.05, π
t = 0.0:0.05:500.0
M = vcat(sin.(√ϕ₁ * t') / √ϕ₁, 3sin.(√ϕ₂ * t') / √ϕ₂, cos.(√ϕ₁ * t'), 3cos.(√ϕ₂ * t'))

nn = NeuralNetwork(PODArch(4, 2))
solve!(nn, M)

V = Matrix(params(nn).L1.weight)
isapprox(abs.(V), [1 0; 0 0; 0 0; 0 1]; atol = 1e-2)

# output

true
```
"""
struct PODArch <: AutoEncoder
    full_dim::Int
    reduced_dim::Int
    n_encoder_blocks::Int
    n_decoder_blocks::Int
end

function PODArch(full_dim::Integer, reduced_dim::Integer)
    @assert full_dim ≥ reduced_dim "Full order dim has to be greater than reduced order dim!"
    PODArch(full_dim, reduced_dim, 2, 2)
end

function encoder_layers_from_iteration(arch::PODArch, encoder_iterations::AbstractVector{<:Integer})
    @assert length(encoder_iterations) == 2
    @assert arch.full_dim == encoder_iterations[1]
    @assert arch.reduced_dim == encoder_iterations[2]

    (StiefelLayer(arch.full_dim, arch.reduced_dim),)
end

function decoder_layers_from_iteration(arch::PODArch, decoder_iterations::AbstractVector{<:Integer})
    @assert length(decoder_iterations) == 2
    @assert arch.full_dim == decoder_iterations[2]
    @assert arch.reduced_dim == decoder_iterations[1]

    (StiefelLayer(arch.reduced_dim, arch.full_dim),)
end

@doc raw"""
    solve!(nn::NeuralNetwork{<:PODArch}, input)

Compute the POD basis of a snapshot matrix `input` and store it in the encoder and the decoder of `nn`.

[`PODArch`](@ref) does not require neural network training: the basis consists of the leading `nn.architecture.reduced_dim` left singular vectors of `input`, obtained with singular value decomposition (SVD).
Returns the [`AutoEncoderLoss`](@ref) of `nn` on `input`.
"""
function solve!(nn::NeuralNetwork{<:PODArch}, input::AbstractMatrix)
    V = svd(input).U
    @views params(nn)[1].weight.A .= V[:, 1:(nn.architecture.reduced_dim)]
    @views params(nn)[2].weight.A .= V[:, 1:(nn.architecture.reduced_dim)]

    AutoEncoderLoss()(nn, input)
end

function solve!(nn::NeuralNetwork{<:PODArch}, input::AbstractArray{T, 3}) where {T}
    solve!(nn, reshape(input, size(input, 1), size(input, 2) * size(input, 3)))
end

function solve!(nn::NeuralNetwork{<:PODArch},
        dl::DataLoader{T, AT, <:Any, :RegularData}) where {T, AT <: AbstractArray{T}}
    solve!(nn, dl.input)
end

function solve!(nn::NeuralNetwork{<:PODArch},
        dl::DataLoader{T, NT, <:Any, :RegularData}) where {
        T, AT <: AbstractArray{T}, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    solve!(nn, vcat(dl.input.q, dl.input.p))
end
