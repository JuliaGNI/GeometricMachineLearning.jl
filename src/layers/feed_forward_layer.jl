
struct FeedForwardLayer{DT, N, M, ST, WT <: AbstractMatrix{DT}, BT <: AbstractVector{DT}, GT} <: NeuralNetworkLayer{DT,N,M}
    σ::ST
    W::WT
    b::BT
    gradient::GT

    function FeedForwardLayer(σ, W::AbstractMatrix{DT}, b::AbstractVector{DT}, gradient = NullGradient()) where {DT}
        @assert length(axes(W,1)) == length(axes(b,1))
        new{DT, length(axes(W,2)), length(axes(W,1)), typeof(σ), typeof(W), typeof(b), typeof(gradient)}(σ,W,b,gradient)
    end
end

function apply!(output::AbstractVector, input::AbstractVector, layer::FeedForwardLayer)
    mul!(output, layer.W, input)
    add!(output, layer.b)
    output .= layer.σ.(output)
end

function apply(input::AbstractVector, layer::FeedForwardLayer)
    layer.σ.(layer.W * input .+ layer.b)
end

(layer::FeedForwardLayer)(x) = apply(x, layer)

const FeedForwardLayerWT{WT, DT, N, M, ST, BT, GT} = FeedForwardLayer{DT, N, M, ST, WT, BT, GT}
const FeedForwardLayerBT{BT, DT, N, M, ST, WT, GT} = FeedForwardLayer{DT, N, M, ST, WT, BT, GT}
# const FeedForwardLayerZeroBT{DT, N, M, ST, WT, GT} = FeedForwardLayerBT{<: ZeroVector, DT, N, M, ST, WT, GT}

parameters(layer::FeedForwardLayer{DT, N, M, ST, WT, BT, GT}) where {DT, N, M, ST, WT, BT <: AbstractVector{DT}, GT} = (W = layer.W, b = layer.b)
parameters(layer::FeedForwardLayer{DT, N, M, ST, WT, BT, GT}) where {DT, N, M, ST, WT, BT <: ZeroVector{DT}, GT} = (W = layer.W, )


const LinearFeedForwardLayer{DT, N, M, WT <: AbstractMatrix{DT}, BT <: AbstractVector{DT}, GT} = FeedForwardLayer{DT, N, M, <: IdentityActivation, WT, BT, GT}

LinearFeedForwardLayer(W, b, gradient = NullGradient()) = FeedForwardLayer(IdentityActivation(), W, b, gradient)

function apply!(output::AbstractVector, input::AbstractVector, layer::LinearFeedForwardLayer)
    mul!(output, layer.W, input)
    add!(output, layer.b)
end

function apply(input::AbstractVector, layer::LinearFeedForwardLayer)
    layer.W * input .+ layer.b
end
