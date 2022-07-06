
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



function apply!(output::AbstractVector, input::AbstractVector, network::VanillaNeuralNetwork)
    # Implement application of vanilla network
    # ...
end

function apply!(output::AbstractVector, input::AbstractVector, network::Inverse{DT,NNT}) where {DT, NNT <: VanillaNeuralNetwork{DT}}
    # Implement application of inverse vanilla network
    # ...
end

function train!(network::VanillaNeuralNetwork, data)
    # Implement training of vanilla network
    # ...
end
