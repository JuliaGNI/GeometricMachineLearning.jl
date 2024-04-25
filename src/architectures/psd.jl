struct PSDArch <: SymplecticCompression 
    full_dim::Int 
    reduced_dim::Int 
    n_encoder_blocks::Int 
    n_decoder_blocks::Int
end

function PSDArch(full_dim::Integer, reduced_dim::Integer)
    @assert iseven(full_dim) && iseven(reduced_dim) "Full order and reduced dimension have to be even!"
    PSDArch(full_dim, reduced_dim, 2, 2)
end

function encoder_layers_from_iteration(arch::PSDArch, encoder_iterations::AbstractVector{<:Integer})
    @assert length(encoder_iterations) == 2
    @assert arch.full_dim == encoder_iterations[1]
    @assert arch.reduced_dim == encoder_iterations[2]

    (PSDLayer(arch.full_dim, arch.reduced_dim), )
end

function decoder_layers_from_iteration(arch::PSDArch, decoder_iterations::AbstractVector{<:Integer})
    @assert length(decoder_iterations) == 2
    @assert arch.full_dim == decoder_iterations[2]
    @assert arch.reduced_dim == decoder_iterations[1]

    (PSDLayer(arch.reduced_dim, arch.full_dim), )
end

# this performs PSD 
function solve!(nn::NeuralNetwork{<:PSDArch}, input::AbstractMatrix)
    half_of_dimension_in_big_space = nn.architecture.full_dim ÷ 2
    @views input_qp = hcat(input[1 : half_of_dimension_in_big_space, :], input[(half_of_dimension_in_big_space + 1) : end, :])
    U_term = svd(input_qp).U
    @views nn.params[1].weight.A .= U_term[:, 1 : nn.architecture.reduced_dim ÷ 2]
    @views nn.params[2].weight.A .= U_term[:, 1 : nn.architecture.reduced_dim ÷ 2]

    AutoEncoderLoss()(nn, input)
end

function solve!(nn::NeuralNetwork{<:PSDArch}, input::AbstractArray{T, 3}) where T 
    solve!(nn, reshape(input, size(input, 1), size(input, 2) * size(input, 3)))
end

function solve!(nn::NeuralNetwork{<:PSDArch}, dl::DataLoader{T, AT, <:Any, :RegularData}) where {T, AT <: AbstractArray{T}}
    solve!(nn, dl.input)
end

function solve!(nn::NeuralNetwork{<:PSDArch}, dl::DataLoader{T, NT, <:Any, :RegularData}) where {T, AT <: AbstractArray{T}, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    solve!(nn, vcat(dl.input.q, dl.input.p))
end