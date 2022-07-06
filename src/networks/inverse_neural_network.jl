
struct Inverse{DT, NNT <: AbstractNeuralNetwork{DT}} <: AbstractNeuralNetwork{DT}
    network::NNT

    function Inverse(network::NNT) where {DT, NNT <: AbstractNeuralNetwork{DT}}
        new{DT,NNT}(network)
    end
end


function apply!(::AbstractVector, ::AbstractVector, ::Inverse{DT,NNT}) where {DT,NNT}
    error("Inverse not supported by network type ", NNT)
end
