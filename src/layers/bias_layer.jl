@doc raw"""
A *bias layer* that does nothing more than add a vector to the input. This is needed for *LA-SympNets*.
"""
struct BiasLayer{M, N} <: SympNetLayer{M, N}
end

function BiasLayer(M::Int)
    BiasLayer{M, M}()
end

function initialparameters(backend::Backend, ::Type{T}, ::BiasLayer{M, M}; rng::AbstractRNG = Random.default_rng(), init_bias = GlorotUniform()) where {M, T}
    q_part = KernelAbstractions.zeros(backend, T, M÷2)
    p_part = KernelAbstractions.zeros(backend, T, M÷2)
    init_bias(rng, q_part)
    init_bias(rng, p_part)
    return (q = q_part, p = p_part)
end

function parameterlength(::BiasLayer{M, M}) where M
    M 
end

(::BiasLayer{M, M})(z::NT, ps::NT) where {M, AT<:AbstractVector, NT<:NamedTuple{(:q, :p), Tuple{AT, AT}}} =  (q = z.q + ps.q, p = z.p + ps.p)
(::BiasLayer{M, M})(z::NT1, ps::NT2) where {M, T, AT<:AbstractVector, BT<:Union{AbstractMatrix, AbstractArray{T, 3}}, NT1<:NamedTuple{(:q, :p), Tuple{AT, AT}}, NT2<:NamedTuple{(:q, :p), Tuple{BT, BT}}} =  (q = z.q .+ ps.q, p = z.p .+ ps.p)

function (d::BiasLayer{M, M})(z::AbstractArray, ps) where M
    apply_layer_to_nt_and_return_array(z, d, ps)
end