"""
MultiHeadAttention (MHA) serves as a preprocessing step in the transformer. It reweights the input vectors bases on correlations within those data. 
"""
struct Attention{M, N, Stiefel, Retraction, add_connection, FT} <: AbstractExplicitLayer{M, N}
    activation::FT
end

default_retr = Geodesic()
function orthonormal_activation(A::AbstractMatrix{T}) where T 
    exp(A - A')
end
# TODO: This can be implemented more efficiently if you write one kernel for everything!
function orthonormal_activation(A::AbstractArray{T, 3}) where T 
    tensor_exponential(T(.5)*(A - tensor_transpose(A)))
end

function Attention(dim::Integer, activation=orthonormal_activation; Stiefel::Bool=false, retraction::AbstractRetraction=default_retr, add_connection::Bool=true)
    Attention{dim, dim, Stiefel, typeof(retraction), add_connection, typeof(activation)}(activation)
end

function parameterlength(::Attention{M, M, false}) where M
    2*M^2
end

function parameterlength(d::Attention{M, M, true}) where M
    M*(M-1)
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, d::Attention{M, M, false}; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    # transformations for queries and keys.
    PQ_weight = KernelAbstractions.allocate(backend, T, M, M)
    PK_weight = KernelAbstractions.allocate(backend, T, M, M)
    initializer(rng, PQ_weight)
    initializer(rng, PK_weight)
    (PQ=PQ_weight, PK=PK_weight)
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, d::Attention{M, M, true}; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    # projections for queries, keys and vectors.
    PQ_weight = rand(backend, rng, StiefelManifold{T}, M, M)
    PK_weight = rand(backend, rng, StiefelManifold{T}, M, M)
    (PQ=PQ_weight, PK=PK_weight)
end

function (d::Attention{M, M, Stiefel, Retraction, true})(x::AbstractMatrix{T}, ps::NamedTuple) where {M, Stiefel, Retraction, T}
    dim, input_length = size(x)
    @assert dim == M

    x + x*d.activation((ps.PQ*x)'*(ps.PK*x))
end

function (d::Attention{M, M, Stiefel, Retraction, false})(x::AbstractMatrix{T}, ps::NamedTuple) where {M, Stiefel, Retraction, T}
    dim, input_length = size(x)
    @assert dim == M

    x*d.activation((ps.PQ*x)'*(ps.PK*x))
end

function (d::Attention{M, M, Stiefel, Retraction, true})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, Stiefel, Retraction, T} 
    dim, input_length, number_data = size(x)
    @assert dim == M

    Q_tensor = mat_tensor_mul(ps.PQ', x)
    K_tensor = mat_tensor_mul(ps.PK', x)
    QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)
    x + tensor_tensor_mul(x, d.activation(QK_tensor))
end

function (d::Attention{M, M, Stiefel, Retraction, false})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, Stiefel, Retraction, T} 
    dim, input_length, number_data = size(x)
    @assert dim == M

    Q_tensor = mat_tensor_mul(ps.PQ', x)
    K_tensor = mat_tensor_mul(ps.PK', x)
    QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)
    tensor_tensor_mul(x, d.activation(QK_tensor))
end