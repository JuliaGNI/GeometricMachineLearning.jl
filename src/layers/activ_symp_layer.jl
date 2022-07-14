#residual layer that changes p
struct SympActLayer_p{DT, N, M, ST, AT <: AbstractVector{DT}, GT} <: NeuralNetworkLayer{DT,N,M}
    a::AT
    σ::ST
    gradient::GT

    function SympActLayer_p(σ, a::AbstractVector{DT}, gradient) where {DT}
        new{DT, length(a), length(a), typeof(a), typeof(gradient)}(a,σ,gradient)
    end
end

function apply!(output::AbstractVector, input::AbstractVector, layer::SympActLayer_p)
    q = @view input[1:N÷2]
    p = @view input[N÷2+1:N]
    p_out = @view output[N÷2+1:N] 
    p_out .= a.*sigma.(q) .+ p
end


#residual layer that changes q
struct SympActLayer_p{DT, N, M, ST, AT <: AbstractVector{DT}, GT} <: NeuralNetworkLayer{DT,N,M}
    a::AT
    σ::ST
    gradient::GT

    function SympActLayer_q(σ, a::AbstractVector{DT}, gradient) where {DT}
        new{DT, length(a), length(a), typeof(a), typeof(gradient)}(a,σ,gradient)
    end
end

function apply!(output::AbstractVector, input::AbstractVector, layer::SympActLayer_p)
    q = @view input[1:N÷2]
    p = @view input[N÷2+1:N]
    q_out = @view output[1:N÷2] 
    q_out .= a.*sigma.(p) .+ q
end



