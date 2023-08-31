"""
This is a ResNet layer, not a residual layer!!!! should be removed anyway!!!
"""


struct ResidualLayer{DT, N, M, ST, WT <: AbstractMatrix{DT}, BT <: AbstractVector{DT}, GT} <: NeuralNetworkLayer{DT,N,M}
    σ::ST
    W::WT
    b::BT
    gradient::GT

    function ResidualLayer(σ, W::AbstractMatrix{DT}, b::AbstractVector{DT}, gradient = NullGradient()) where {DT}
        @assert length(axes(W,1)) == length(axes(b,1))
        new{DT, length(axes(W,2)), length(axes(W,1)), typeof(σ), typeof(W), typeof(b), typeof(gradient)}(σ,W,b,gradient)
    end
end

function apply!(output::AbstractVector, input::AbstractVector, layer::ResidualLayer)
    mul!(output, layer.W, input)
    add!(output, layer.b)
    output .= layer.σ.(output)
    output .+= input
end

const ResidualLayerWT{WT, DT, N, M, ST, BT, GT} = ResidualLayer{DT, N, M, ST, WT, BT, GT}
const ResidualLayerBT{BT, DT, N, M, ST, WT, GT} = ResidualLayer{DT, N, M, ST, WT, BT, GT}
# const ResidualLayerZeroBT = ResidualLayerBT{BT} where {BT <: ZeroVector}

parameters(layer::ResidualLayer{DT, N, M, ST, WT, BT, GT}) where {DT, N, M, ST, WT, BT <: AbstractVector{DT}, GT} = (W = layer.W, b = layer.b)
parameters(layer::ResidualLayer{DT, N, M, ST, WT, BT, GT}) where {DT, N, M, ST, WT, BT <: ZeroVector{DT}, GT} = (W = layer.W, )
