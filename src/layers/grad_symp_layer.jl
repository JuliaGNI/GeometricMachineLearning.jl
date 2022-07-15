#residual layer that changes p
struct SympActLayer_p{DT, N, M, ST, KT <: AbstractMatrix{DT}, BT <: AbstractVector{DT}, GT} <: NeuralNetworkLayer{DT,N,M}
    σ::ST
    K::KT
    b::BT
    gradient::GT

    function SympActLayer_p(σ, K::AbstractMatrix{DT}, b::AbstractVector{DT}, gradient) where {DT}
        new{DT, 2*length(b), 2*length(b), typeof(σ), typeof(K), typeof(b), typeof(gradient)}(σ,K,b,gradient)
    end
end

function apply!(output::AbstractVector, input::AbstractVector, layer::SympActLayer_p)
    #get rid of this?
    N = 2*length(layer.b)
    q = @view input[1:N÷2]
    p = @view input[N÷2+1:N]
    p_out = @view output[N÷2+1:N] 
    p_out .= layer.K'*layer.σ.(K*q .+ b) .+ p
end


#residual layer that changes q
struct SympActLayer_q{DT, N, M, ST, KT <: AbstractMatrix{DT}, BT <: AbstractVector{DT}, GT} <: NeuralNetworkLayer{DT,N,M}
    σ::ST
    K::KT
    b::BT
    gradient::GT

    function SympActLayer_q(σ, K::AbstractMatrix{DT}, b::AbstractVector{DT}, gradient) where {DT}
        new{DT, 2*length(b), 2*length(b), typeof(σ), typeof(K), typeof(b), typeof(gradient)}(σ,K,b,gradient)
    end
end

function apply!(output::AbstractVector, input::AbstractVector, layer::SympActLayer_q)
    #get rid of this?
    N = 2*length(layer.b)
    q = @view input[1:N÷2]
    p = @view input[N÷2+1:N]
    q_out = @view output[N÷2+1:N] 
    q_out .= layer.K'*layer.σ.(K*p .+ b) .+ q
end


