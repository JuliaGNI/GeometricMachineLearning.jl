struct PSDArch <: AutoEncoder 
    full_dim::Int 
    reduced_dim::Int 
    n_encoder_blocks::Int 
    n_decoder_blocks::Int
end

function PSDArch(full_dim::Integer, reduced_dim::Integer)
    PSDArch(full_dim, reduced_dim, 1, 1)
end

function Chain(arch::PSDArch)
    Chain(Linear(arch.full_dim, arch.reduced_dim; use_bias = false), Linear(arch.reduced_dim, arch.full_dim; use_bias = false))
end

#=
get_ecoder(arch::PSDArch) = Linear(arch.full_dim, arch.reduced_dim)

get_decoder(arch::PSDArch) = Linear(arch.reduced_dim, arch.full_dim)

get_encoder_parameters(nn::NeuralNetwork{<:PSDArch}) = nn.params[1]

get_decoder_parameters(nn::NeuralNetwork{<:PSDArch}) = nn.params[2]
=#

function solve!(nn::NeuralNetwork{<:PSDArch}, dl::DataLoader{T, AT, <:Any, :RegularData}) where {T, AT <: AbstractArray{T}}
    #nn dl.input
end

function solve!(nn::NeuralNetwork{<:PSDArch}, dl::DataLoader{T, NT, <:Any, :RegularData}) where {T, AT <: AbstractArray{T}, NT <: NamedTuple{(:q, :p), {AT, AT}}}

end