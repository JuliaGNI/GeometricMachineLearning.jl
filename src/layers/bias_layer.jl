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

function (::BiasLayer{M, M})(z::NT, ps::NT) where {
                                                    M, 
                                                    AT<:AbstractVector, 
                                                    NT<:NamedTuple{(:q, :p), Tuple{AT, AT}}
                                                    }
    (q = z.q + ps.q, p = z.p + ps.p)
end

function (::BiasLayer{M, M})(z::NT2, ps::NT1) where {   
                                                    M, 
                                                    T, 
                                                    AT<:AbstractVector{T}, 
                                                    BT<:Union{AbstractMatrix{T}, AbstractArray{T, 3}}, 
                                                    NT1<:NamedTuple{(:q, :p), Tuple{AT, AT}}, 
                                                    NT2<:NamedTuple{(:q, :p), Tuple{BT, BT}}
                                                    }
    (q = z.q .+ ps.q, p = z.p .+ ps.p)
end

function (d::BiasLayer{M, M})(z::AbstractArray, ps) where M
    apply_layer_to_nt_and_return_array(z, d, ps)
end