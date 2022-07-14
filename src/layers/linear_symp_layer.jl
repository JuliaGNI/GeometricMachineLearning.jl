#residual layer that changes p
struct ResidualLayer_p{DT, N, M, WT <: AbstractMatrix{DT}, GT} <: NeuralNetworkLayer{DT,N,M}
    W::WT
    S::WT
    gradient::GT

    function ResidualLayer(W::AbstractMatrix{DT}, gradient) where {DT}
        @assert length(axes(W,1)) == length(axes(W,2))
        S = W .+ W'
        new{DT, length(axes(W,2)), length(axes(W,1)), typeof(W), typeof(gradient)}(W,S,gradient)
    end
end

function apply!(output::AbstractVector, input::AbstractVector, layer::ResidualLayer)
    q = @view input[1:N÷2]
    p = @view input[N÷2+1:N]
    p_out = @view output[N÷2+1:N] 
    mul!(p_out, S, q)
    p_out .+= p
end


#residual layer that changes q
struct ResidualLayer_q{DT, N, M, WT <: AbstractMatrix{DT}, GT} <: NeuralNetworkLayer{DT,N,M}
    W::WT
    S::WT
    gradient::GT

    function ResidualLayer(W::AbstractMatrix{DT}, gradient) where {DT}
        @assert length(axes(W,1)) == length(axes(W,2))
        S = W .+ W'
        new{DT, length(axes(W,2)), length(axes(W,1)), typeof(W), typeof(gradient)}(W,S,gradient)
    end
end

function apply!(output::AbstractVector, input::AbstractVector, layer::ResidualLayer)
    q = @view input[1:N÷2]
    p = @view input[N÷2+1:N]
    q_out = @view output[N÷2+1:N] 
    mul!(q_out, S, q)
    q_out .+= q
end


