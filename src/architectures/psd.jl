@doc raw"""
    PSDArch <: SymplecticCompression

`PSDArch` is the architecture that corresponds to [`PSDLayer`](@ref). 

## The architecture 

Proper symplectic decomposition (PSD) can be seen as a [`SymplecticAutoencoder`](@ref) for which the decoder and the encoder are both PSD-like matrices (see the docs for [`PSDLayer`](@ref). 

## Training 

For optimizing the parameters in this architecture no neural network training is necessary (see the docs for [`solve!`](@ref)).

## The constructor 

The constructor only takes two arguments as input:
- `full_dim::Integer`
- `reduced_dim::Integer`

with `full_dim > reduced_dim >= 1` and both even; otherwise it throws an `ArgumentError`.
"""
struct PSDArch <: SymplecticCompression
    full_dim::Int
    reduced_dim::Int
    n_encoder_blocks::Int
    n_decoder_blocks::Int
end

function PSDArch(full_dim::Integer, reduced_dim::Integer)
    reduced_dim >= 1 ||
        throw(ArgumentError("reduced_dim ($reduced_dim) must be at least 1."))
    full_dim > reduced_dim ||
        throw(ArgumentError("full_dim ($full_dim) must be greater than reduced_dim ($reduced_dim)."))
    iseven(full_dim) && iseven(reduced_dim) ||
        throw(ArgumentError("full_dim ($full_dim) and reduced_dim ($reduced_dim) must be even."))
    PSDArch(full_dim, reduced_dim, 2, 2)
end

function encoder_layers_from_iteration(arch::PSDArch, encoder_iterations::AbstractVector{<:Integer})
    @assert length(encoder_iterations) == 2
    @assert arch.full_dim == encoder_iterations[1]
    @assert arch.reduced_dim == encoder_iterations[2]

    (PSDLayer(arch.full_dim, arch.reduced_dim),)
end

function decoder_layers_from_iteration(arch::PSDArch, decoder_iterations::AbstractVector{<:Integer})
    @assert length(decoder_iterations) == 2
    @assert arch.full_dim == decoder_iterations[2]
    @assert arch.reduced_dim == decoder_iterations[1]

    (PSDLayer(arch.reduced_dim, arch.full_dim),)
end

@doc raw"""
    solve!(nn::NeuralNetwork{<:PSDArch}, input)

Solve the cotangent lift problem for an input.

[`PSDArch`](@ref) does not require neural network training since it is a strictly linear operation that can be solved with singular value decomposition (SVD).
The SVD is that of the cotangent lift ``[q\ p]``, which has two columns per snapshot of `input`; it must have at least `reduced_dim ÷ 2` columns, so `input` needs at least `reduced_dim ÷ 4` snapshots, rounded up. Otherwise `solve!` throws an `ArgumentError`.
"""
function solve!(nn::NeuralNetwork{<:PSDArch}, input::AbstractMatrix)
    n_half = nn.architecture.reduced_dim ÷ 2
    2 * size(input, 2) >= n_half ||
        throw(ArgumentError("the cotangent lift of input has $(2 * size(input, 2)) columns, fewer than reduced_dim ÷ 2 ($n_half)."))
    half_of_dimension_in_big_space = nn.architecture.full_dim ÷ 2
    @views input_qp = hcat(input[1:half_of_dimension_in_big_space, :],
        input[(half_of_dimension_in_big_space + 1):end, :])
    U_term = svd(input_qp).U
    @views params(nn)[1].weight.A .= U_term[:, 1:(nn.architecture.reduced_dim ÷ 2)]
    @views params(nn)[2].weight.A .= U_term[:, 1:(nn.architecture.reduced_dim ÷ 2)]

    AutoEncoderLoss()(nn, input)
end

# The SVD of each architecture is in its matrix method; the other inputs reduce to a matrix.
function solve!(nn::NeuralNetwork{<:Union{PSDArch, PODArch}},
        input::AbstractArray{T, 3}) where {T}
    solve!(nn, reshape(input, size(input, 1), size(input, 2) * size(input, 3)))
end

function solve!(nn::NeuralNetwork{<:Union{PSDArch, PODArch}},
        dl::DataLoader{T, AT, <:Any, :RegularData}) where {T, AT <: AbstractArray{T}}
    solve!(nn, dl.input)
end

function solve!(nn::NeuralNetwork{<:Union{PSDArch, PODArch}},
        dl::DataLoader{T, NT, <:Any, :RegularData}) where {
        T, AT <: AbstractArray{T}, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    solve!(nn, vcat(dl.input.q, dl.input.p))
end
