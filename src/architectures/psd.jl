struct PSD <: AutoEncoder 
    full_dim::Int 
    reduced_dim::Int 
    n_encoder_blocks::Int 
    n_decoder_blocks::Int
end

function PSD(full_dim::Integer, reduced_dim::Integer)
    PSD(full_dim, reduced_dim, 1, 1)
end

function Chain(arch::PSD)
    Chain(Linear(arch.full_dim, arch.reduced_dim; use_bias = false), Linear(arch.reduced_dim, arch.full_dim; use_bias = false))
end

#=
get_ecoder(arch::PSD) = Linear(arch.full_dim, arch.reduced_dim)

get_decoder(arch::PSD) = Linear(arch.reduced_dim, arch.full_dim)

get_encoder_parameters(nn::NeuralNetwork{<:PSD}) = nn.params[1]

get_decoder_parameters(nn::NeuralNetwork{<:PSD}) = nn.params[2]
=#

function solve!(nn::NeuralNetwork{<:PSD}, dl::DataLoader{T, AT, <:Any, :RegularData}) where {T, AT <: AbstractArray{T}}
    #nn dl.input
end

function solve!(nn::NeuralNetwork{<:PSD}, dl::DataLoader{T, NT, <:Any, :RegularData}) where {T, AT <: AbstractArray{T}, NT <: NamedTuple{(:q, :p), {AT, AT}}}

end