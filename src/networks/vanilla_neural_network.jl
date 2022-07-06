
abstract type NeuralNetworkType end

struct VariableWidthNetwork <: NeuralNetworkType end
struct FixedWidthNetwork <: NeuralNetworkType end
struct AutoEncoderNetwork <: NeuralNetworkType end


struct VanillaNeuralNetwork{DataType <: Number, NetworkType <: NeuralNetworkType, LayersType <: Tuple} <: AbstractNeuralNetwork{DataType}
    layers::LayersType

    function VanillaNeuralNetwork{DT}(nn_type::NeuralNetworkType, layers...) where {DT}
        new{DT, typeof(nn_type), typeof(layers)}(layers)
    end
end


function apply!(output::AbstractVector, input::AbstractVector, network::VanillaNeuralNetwork{DT}) where {DT}
    temp = merge( input, ( zeros(DT, output_size(layer)) for layer in network.layers) )

    for i in eachindex(network.layers)
        apply!(temp[i+1], temp[i], network.layers[i])
    end

    return output .= temp[end]
end

function apply!(output::AbstractVector, input::AbstractVector, network::VanillaNeuralNetwork{DT,NNT}) where {DT, NNT <: FixedWidthNetwork}
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
