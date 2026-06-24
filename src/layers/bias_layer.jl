# A *bias layer* that does nothing more than add a vector to the input. This is needed for *LA-SympNets*.
struct BiasLayer{M, N} <: SympNetLayer{M, N}
end

function BiasLayer(M::Int)
    BiasLayer{M, M}()
end

function initialparameters(rng::AbstractRNG, init_bias::AbstractNeuralNetworks.Initializer, ::BiasLayer{M, M}, backend::Backend, ::Type{T}) where {M, T}
    q_part = KernelAbstractions.zeros(backend, T, M÷2)
    p_part = KernelAbstractions.zeros(backend, T, M÷2)
    init_bias(rng, q_part)
    init_bias(rng, p_part)
    return (q = q_part, p = p_part)
end

function parameterlength(::BiasLayer{M, M}) where M
    M 
end

function (::BiasLayer{M, M})(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {
                                                    M,
                                                    AT<:AbstractVector
                                                    }
    (q = z.q + ps.q, p = z.p + ps.p)
end

function (::BiasLayer{M, M})(z::NamedTuple{(:q, :p), Tuple{BT, BT}}, ps::NamedTuple) where {
                                                    M,
                                                    BT<:Union{AbstractMatrix, AbstractArray{<:Any, 3}}
                                                    }
    (q = z.q .+ ps.q, p = z.p .+ ps.p)
end

function (d::BiasLayer{M, M})(z::AbstractArray, ps::NamedTuple) where M
    apply_layer_to_nt_and_return_array(z, d, ps)
end