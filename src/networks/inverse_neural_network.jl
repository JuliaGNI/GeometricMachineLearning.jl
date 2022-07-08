
struct Inverse{NNT <: AbstractNeuralNetwork} <: AbstractNeuralNetwork
    network::NNT

    function Inverse(network::NNT) where {NNT <: AbstractNeuralNetwork}
        new{NNT}(network)
    end
end


function apply!(::AbstractVector, ::AbstractVector, ::Inverse{NNT}) where {NNT}
    error("Inverse not supported by network type ", NNT)
end
