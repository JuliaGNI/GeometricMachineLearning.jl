@doc raw"""
    SymplecticAttention

Implements the symplectic attention layers. See [`LinearSymplecticAttention`](@ref).
"""
struct SymplecticAttention{M, N, LayerType, Symmetric} <: AbstractExplicitLayer{M, N} 
end

const SymplecticAttentionQ{M, N, Symmetric} = SymplecticAttention{M, N, :Q, Symmetric}

const SymplecticAttentionP{M, N, Symmetric} = SymplecticAttention{M, N, :P, Symmetric}

function SymplecticAttentionQ(M::Integer; symmetric = false)
    @assert iseven(M) "Dimension must be even!"
    symmetric == false ? SymplecticAttention{M, M, :Q, :arbitrary}() : SymplecticAttention{M, M, :Q, :symmetric}()
end
function SymplecticAttentionP(M::Integer; symmetric = false)
    @assert iseven(M) "Dimension must be even!"
    symmetric == false ? SymplecticAttention{M, M, :P, :arbitrary}() : SymplecticAttention{M, M, :P, :symmetric}()
end
 
function parameterlength(::SymplecticAttention{M, M, LayerType, :arbitrary})::Integer where {M, LayerType} 
    M2 = M ÷ 2
    M2 * M2
end

function parameterlength(::SymplecticAttention{M, M, LayerType, :symmetric})::Integer where {M, LayerType}
    M2 = M ÷ 2
    (M2 + 1) * M2 ÷ 2
end

function initialparameters( rng::AbstractRNG, 
                            initializer::AbstractNeuralNetworks.Initializer, 
                            l::SymplecticAttention{M, M, LayerType, :symmetric}, 
                            backend::KernelAbstractions.Backend, 
                            T::Type) where {M, LayerType}
    S = KernelAbstractions.allocate(backend, T, parameterlength(l))
    initializer(rng, S)
    (A = SymmetricMatrix(S, M ÷ 2), )
end

function initialparameters( rng::AbstractRNG, 
                            initializer::AbstractNeuralNetworks.Initializer, 
                            ::SymplecticAttention{M, M, LayerType, :arbitrary}, 
                            backend::KernelAbstractions.Backend, 
                            T::Type) where {M, LayerType}
    A = KernelAbstractions.allocate(backend, T, M ÷ 2, M ÷ 2)
    initializer(rng, A)
    (A = A, )
end

function (::SymplecticAttentionQ{M, M, :arbitrary})(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M}
    expPAP = exp.(_custom_mul(_custom_mul(_custom_transpose(z.p), ps.A), z.p))
    (q = z.q + (_custom_mul(_custom_mul(ps.A, z.p), _custom_transpose(expPAP)) + _custom_mul(_custom_mul(ps.A', z.p), expPAP)) / sum(expPAP), p = z.p)
end

function (::SymplecticAttentionQ{M, M, :symmetric})(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M}
    expPAP = exp.(_custom_mul(_custom_mul(_custom_transpose(z.p), ps.A), z.p))
    (q = z.q + _custom_mul(_custom_mul(ps.A, z.p), 2 * expPAP) / sum(expPAP), p = z.p)
end

function (::SymplecticAttentionP{M, M, :arbitrary})(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M}
    expQAQ = exp.(_custom_mul(_custom_mul(_custom_transpose(z.q), ps.A), z.q))
    (q = z.q, p = z.p + (_custom_mul(_custom_mul(ps.A, z.q), _custom_transpose(expQAQ)) + _custom_mul(_custom_mul(ps.A', z.q), expQAQ)) / sum(expQAQ))
end

function (::SymplecticAttentionP{M, M, :symmetric})(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, ps::NamedTuple) where {AT, M}
    expQAQ = exp.(_custom_mul(_custom_mul(_custom_transpose(z.q), ps.A), z.q))
    (q = z.q, p = z.p + _custom_mul(_custom_mul(ps.A, z.q), 2 * expQAQ) / sum(expQAQ))
end

function (d::SymplecticAttention)(z::AbstractArray, ps::NamedTuple)
    apply_layer_to_nt_and_return_array(z, d, ps)
end

_custom_mul(A::AbstractMatrix{T}, z::AbstractArray{T, 3}) where T = mat_tensor_mul(A, z)
_custom_mul(Z1::AbstractArray{T, 3}, Z2::AbstractArray{T, 3}) where T = tensor_tensor_mul(Z1, Z2)

_custom_transpose(Z::AbstractMatrix) = Z'
_custom_transpose(z::AbstractArray{T, 3}) where T = tensor_transpose(z)