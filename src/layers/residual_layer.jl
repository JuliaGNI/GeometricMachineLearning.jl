
struct ResidualLayer{DT,N,M,ST,WT,BT} <: NeuralNetworkLayer{DT,N,M}
    σ::ST
    W::WT
    b::BT

    function ResidualLayer(σ, W::AbstractMatrix{DT}, b::AbstractVector{DT}) where {DT}
        @assert length(axes(W,1)) == lenght(axes(b,1))
        new{DT, length(axes(W,2)), length(axes(W,1)), typeof(σ), typeof(W), typeof(b)}
    end
end

function apply!(output::AbstractVector, input::AbstractVector, layer::ResidualLayer)
    mul!(output, layer.W, input)
    output .+= layer.b
    output .= layer.σ.(output)
    output .+= input
end
