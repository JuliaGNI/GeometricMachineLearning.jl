
using Zygote: Params

abstract type NeuralNetworkType end

struct VariableWidthNetwork <: NeuralNetworkType end
struct FixedWidthNetwork <: NeuralNetworkType end
struct AutoEncoderNetwork <: NeuralNetworkType end

@generated function _applychain(layers::Tuple{Vararg{<:Any,N}}, x) where {N}
    symbols = vcat(:x, [gensym() for _ in 1:N])
    calls = [:($(symbols[i+1]) = layers[$i]($(symbols[i]))) for i in 1:N]
    Expr(:block, calls...)
end
  
_applychain(layers::NamedTuple, x) = _applychain(Tuple(layers), x)


struct VanillaNeuralNetwork{NetworkType <: NeuralNetworkType, LayersType <: Tuple, ModelType, LossType, GradientType} <: AbstractNeuralNetwork
    layers::LayersType
    model::ModelType
    loss::LossType
    gradient::GradientType

    function VanillaNeuralNetwork(nn_type::NeuralNetworkType, model, loss, gradient, layers...)
        new{typeof(nn_type), typeof(layers), typeof(model), typeof(loss), typeof(gradient)}(layers, model, loss, gradient)
    end
end

function VanillaNeuralNetwork(nn_type::NeuralNetworkType, loss, gradient, layers...)
    model(x) = _applychain(layers, x)
    VanillaNeuralNetwork(nn_type, model, loss, gradient, layers...)
end

apply(input::AbstractVector, network::VanillaNeuralNetwork) = network.model(input)

(network::VanillaNeuralNetwork)(x) = apply(x, network)


@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)

function parameters(network::VanillaNeuralNetwork)
    # Params( parameters(layer) for layer in network.layers )
    paramtuples = ( Tuple(parameters(layer)) for layer in network.layers)

    params = ()

    for p in paramtuples
        params = tuplejoin(params, p)
    end

    Params(params)
end

function apply!(output::AbstractVector{DT}, input::AbstractVector, network::VanillaNeuralNetwork) where {DT}
    temp = merge( input, ( zeros(DT, output_size(layer)) for layer in network.layers) )

    for i in eachindex(network.layers)
        apply!(temp[i+1], temp[i], network.layers[i])
    end

    return output .= temp[end]
end

function apply!(output::AbstractVector, input::AbstractVector, network::VanillaNeuralNetwork{NNT}) where {NNT <: FixedWidthNetwork}
    @assert length(axes(input)) == length(axes(output))

    temp = zero(output)
    output .= input

    for layer in network.layers
        temp .= output
        apply!(output, temp, layer)
    end

    return output
end

# function apply!(output::AbstractVector, input::AbstractVector, network::Inverse{DT,NNT}) where {DT, NNT <: VanillaNeuralNetwork{DT}}
    # Implement application of inverse vanilla network
    # ...
# end

function train!(network::VanillaNeuralNetwork, data)
    # Implement training of vanilla network
    # ...
end
