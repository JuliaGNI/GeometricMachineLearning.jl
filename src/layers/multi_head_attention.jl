"""
MultiHeadAttention (MHA) serves as a preprocessing step in the transformer. It reweights the input vectors bases on correlations within those data. 
"""
struct MultiHeadAttention{M, N, Stiefel, Retraction, add_connection} <: AbstractExplicitLayer{M, N}
    n_heads::Integer
end

default_retr = Geodesic()
function MultiHeadAttention(dim::Integer, n_heads::Integer; Stiefel::Bool=false, retraction::AbstractRetraction=default_retr, add_connection::Bool=true)
    @assert dim % n_heads == 0
    MultiHeadAttention{dim, dim, Stiefel, typeof(retraction), add_connection}(n_heads)
end

function parameterlength(::MultiHeadAttention{M, M, false}) where M
    3*M^2
end

function parameterlength(d::MultiHeadAttention{M, M, true}) where M
    3*M*(2*M*d.n_heads - d.n_heads - M)÷(2*d.n_heads^2)
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, d::MultiHeadAttention{M, M, false}; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    # number of "hidden" dimension (dimension of projection) 
    Dₕ = M ÷ d.n_heads
    # projections for queries, keys and vectors.
    PQ = NamedTuple()
    PK = NamedTuple()
    PV = NamedTuple()

    for head in 1:d.n_heads
        key = Symbol("head_"*string(head))
        
        PQ_weight = KernelAbstractions.allocate(backend, T, M, Dₕ)
        PK_weight = KernelAbstractions.allocate(backend, T, M, Dₕ)
        PV_weight = KernelAbstractions.allocate(backend, T, M, Dₕ)
        initializer(rng, PQ_weight)
        initializer(rng, PK_weight)
        initializer(rng, PV_weight)

        PQ = merge(PQ, 
            NamedTuple{(key, )}((PQ_weight, ))
            )
        PK = merge(PK, 
            NamedTuple{(key, )}((PK_weight, ))
            )
        PV = merge(PV, 
            NamedTuple{(key, )}((PV_weight, ))
            )
    end
    (PQ=PQ, PK=PK, PV=PV)
end


function initialparameters(backend::KernelAbstractions.Backend, T::Type, d::MultiHeadAttention{M, M, true}; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    # number of "hidden" dimension (dimension of projection) 
    Dₕ = M ÷ d.n_heads
    # projections for queries, keys and vectors.
    PQ = NamedTuple()
    PK = NamedTuple()
    PV = NamedTuple()

    for head in 1:d.n_heads
        key = Symbol("head_"*string(head))
        
        PQ = merge(PQ, 
            NamedTuple{(key, )}((rand(backend, rng, StiefelManifold{T}, M, Dₕ), ))
            )
        PK = merge(PK, 
            NamedTuple{(key, )}((rand(backend, rng, StiefelManifold{T}, M, Dₕ), ))
            )
        PV = merge(PV, 
            NamedTuple{(key, )}((rand(backend, rng, StiefelManifold{T}, M, Dₕ), ))
            )
    end
    (PQ=PQ, PK=PK, PV=PV)
end

function (d::MultiHeadAttention{M, M, Stiefel, Retraction, true})(x::AbstractMatrix{T}, ps::NamedTuple) where {M, Stiefel, Retraction, T}
    dim, input_length = size(x)
    @assert dim == M

    output = typeof(x)(zeros(T, 0, input_length))
    for i in 1:d.n_heads
        key = Symbol("head_"*string(i))
        output = vcat(output, ps.PV[key]'*x*Lux.softmax((ps.PQ[key]'*x)'*(ps.PK[key]'*x)))
    end
    x + output
end

function (d::MultiHeadAttention{M, M, Stiefel, Retraction, false})(x::AbstractMatrix{T}, ps::NamedTuple) where {M, Stiefel, Retraction, T}
    dim, input_length = size(x)
    @assert dim == M

    output = typeof(x)(zeros(T, 0, input_length))
    for i in 1:d.n_heads
        key = Symbol("head_"*string(i))
        output = vcat(output, ps.PV[key]'*x*Lux.softmax((ps.PQ[key]'*x)'*(ps.PK[key]'*x)))
    end
    output
end

function (d::MultiHeadAttention{M, M, Stiefel, Retraction, true})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, Stiefel, Retraction, T} 
    Dₕ = M ÷ d.n_heads
    dim, input_length, number_data = size(x)
    @assert dim == M
    

    output = similar(x, 0, input_length, number_data)

    # initialize the results of various tensor matrix multiplications
    Q_tensor = similar(x, Dₕ, input_length, number_data)
    K_tensor = similar(x, Dₕ, input_length, number_data)
    V_tensor = similar(x, Dₕ, input_length, number_data)
    QK_tensor = similar(x, input_length, input_length, number_data)

    # this is the result of a single head attention block
    single_head_output = similar(x, Dₕ, input_length, number_data)

    for i in 1:d.n_heads 
        key = Symbol("head_"*string(i))
        Q_tensor = mat_tensor_mul(ps.PQ[key]', x)
        K_tensor = mat_tensor_mul(ps.PK[key]', x)
        V_tensor = mat_tensor_mul(ps.PV[key]', x)
        QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)

        single_head_output = tensor_tensor_mul(V_tensor, Lux.softmax(QK_tensor))
        output = vcat(output, single_head_output) 
        # KernelAbstractions.synchronize(backend)
    end
    x + output
end

function (d::MultiHeadAttention{M, M, Stiefel, Retraction, false})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, Stiefel, Retraction, T} 
    Dₕ = M ÷ d.n_heads
    dim, input_length, number_data = size(x)
    @assert dim == M
    
    output = similar(x, 0, input_length, number_data)

    Q_tensor = similar(x, Dₕ, input_length, number_data)
    K_tensor = similar(x, Dₕ, input_length, number_data)
    V_tensor = similar(x, Dₕ, input_length, number_data)
    QK_tensor = similar(x, input_length, input_length, number_data)
    
    single_head_output = similar(x, Dₕ, input_length, number_data)

    for i in 1:d.n_heads 
        key = Symbol("head_"*string(i))
        Q_tensor = mat_tensor_mul(ps.PQ[key]', x)
        K_tensor = mat_tensor_mul(ps.PK[key]', x)
        V_tensor = mat_tensor_mul(ps.PV[key]', x)
        QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)

        single_head_output = tensor_tensor_mul(V_tensor, Lux.softmax(QK_tensor))
        output = vcat(output, single_head_output) 
        # KernelAbstractions.synchronize(backend)
    end
    output
end

import ChainRules
function ChainRules._adjoint_mat_pullback(y::AbstractArray{T, 3}, proj) where T 
    (NoTangent(), proj(tensor_transpose(y)))
end

function mat_tensor_mul(Y::AT, x::AbstractArray{T, 3}) where {T, BT <: AbstractArray{T}, ST <: StiefelManifold{T, BT}, AT <: Adjoint{T, ST}}
    mat_tensor_mul(Y.parent.A', x)
end

function mat_tensor_mul(Y::StiefelManifold, x::AbstractArray{T, 3}) where T 
    mat_tensor_mul(Y.A, x)
end