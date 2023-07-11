struct MultiHeadAttention{Stiefel, Retraction, add_connection, F1} <: Lux.AbstractExplicitLayer
    dim::Integer
    n_heads::Integer
    init_weight::F1
end

default_retr = Geodesic()
function MultiHeadAttention(dim::Integer, n_heads::Integer; Stiefel::Bool=false, Retraction::AbstractRetraction=default_retr, add_connection::Bool=true, init_weight=Lux.glorot_uniform)
    @assert dim % n_heads == 0
    MultiHeadAttention{Stiefel, typeof(Retraction), add_connection, typeof(init_weight)}(dim, n_heads, init_weight)
end

Lux.parameterlength(d::StiefelLayer{false}) = 3*d.dim^2
Lux.parameterlength(d::StiefelLayer{true}) = 3*d.dim*(2*d.dim*d.n_heads - d.n_heads - d.dim)÷(2*d.n_heads^2)


function Lux.initialparameters(rng::AbstractRNG, d::MultiHeadAttention{false})
    #number of "hidden" dimension (dimension of projection) 
    Dₕ = d.dim ÷ d.n_heads
    #projections for queries, keys and vectors.
    PQ = NamedTuple()
    PK = NamedTuple()
    PV = NamedTuple()

    for head in 1:d.n_heads
        key = Symbol("head_"*string(head))
        
        PQ = merge(PQ, 
            NamedTuple{(key, )}((d.init_weight(rng, d.dim, Dₕ), ))
            )
        PK = merge(PK, 
            NamedTuple{(key, )}((d.init_weight(rng, d.dim, Dₕ), ))
            )
        PV = merge(PV, 
            NamedTuple{(key, )}((d.init_weight(rng, d.dim, Dₕ), ))
            )
    end
    (PQ=PQ, PK=PK, PV=PV)
end


function Lux.initialparameters(rng::AbstractRNG, d::MultiHeadAttention{true})
    #number of "hidden" dimension (dimension of projection) 
    Dₕ = d.dim ÷ d.n_heads
    #projections for queries, keys and vectors.
    PQ = NamedTuple()
    PK = NamedTuple()
    PV = NamedTuple()

    for head in 1:d.n_heads
        key = Symbol("head_"*string(head))
        
        PQ = merge(PQ, 
            NamedTuple{(key, )}((d.init_weight(rng, StiefelManifold, d.dim, Dₕ), ))
            )
        PK = merge(PK, 
            NamedTuple{(key, )}((d.init_weight(rng, StiefelManifold, d.dim, Dₕ), ))
            )
        PV = merge(PV, 
            NamedTuple{(key, )}((d.init_weight(rng, StiefelManifold, d.dim, Dₕ), ))
            )
    end
    (PQ=PQ, PK=PK, PV=PV)
end

function Lux.apply(d::MultiHeadAttention{Stiefel, Retraction, true}, x::AbstractMatrix{T}, ps::NamedTuple, st::NamedTuple) where {Stiefel, Retraction, T}
    Dₕ = d.dim ÷ d.n_heads
    dim, input_length = size(x)
    @assert dim == d.dim

    output = typeof(x)(zeros(T, 0, input_length))
    for i in 1:d.n_heads
        key = Symbol("head_"*string(i))
        output = vcat(output, ps.PV[key]'*x*Lux.softmax((ps.PQ[key]'*x)'*(ps.PK[key]'*x)))
    end
    x + output, st
end

function Lux.apply(d::MultiHeadAttention{Stiefel, Retraction, false}, x::AbstractMatrix{T}, ps::NamedTuple, st::NamedTuple) where {Stiefel, Retraction, T}
    Dₕ = d.dim ÷ d.n_heads
    dim, input_length = size(x)
    @assert dim == d.dim

    output = typeof(x)(zeros(T, 0, input_length))
    for i in 1:d.n_heads
        key = Symbol("head_"*string(i))
        output = vcat(output, ps.PV[key]'*x*Lux.softmax((ps.PQ[key]'*x)'*(ps.PK[key]'*x)))
    end
    output, st
end

function Lux.apply(d::MultiHeadAttention{Stiefel, Retraction, true}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {Stiefel, Retraction, T} 
    Dₕ = d.dim ÷ d.n_heads
    dim, input_length, number_data = size(x)
    @assert dim == d.dim
    
    #backend = KernelAbstractions.get_backend(x)

    #output = KernelAbstractions.zeros(backend, T, 0, input_length, number_data)
    output = similar(x, 0, input_length, number_data)

    #Q_tensor = KernelAbstractions.zeros(backend, T, Dₕ, input_length, number_data)
    Q_tensor = similar(x, Dₕ, input_length, number_data)
    #K_tensor = KernelAbstractions.zeros(backend, T, Dₕ, input_length, number_data)
    K_tensor = similar(x, Dₕ, input_length, number_data)
    #V_tensor = KernelAbstractions.zeros(backend, T, Dₕ, input_length, number_data)
    V_tensor = similar(x, Dₕ, input_length, number_data)
    #QK_tensor = KernelAbstractions.zeros(backend, T, input_length, input_length, number_data)
    QK_tensor = similar(x, input_length, input_length, number_data)
    #single_head_output = KernelAbstractions.zeros(backend, T, Dₕ, input_length, number_data)
    single_head_output = similar(x, Dₕ, input_length, number_data)

    for i in 1:d.n_heads 
        key = Symbol("head_"*string(i))
        #use tensor_mat_mul for this computation
        #mat_tensor_mul!(Q_tensor, ps.PQ[key]', x)
        Q_tensor = mat_tensor_mul(ps.PQ[key]', x)
        #mat_tensor_mul!(K_tensor, ps.PK[key]', x)
        K_tensor = mat_tensor_mul(ps.PK[key]', x)
        #mat_tensor_mul!(V_tensor, ps.PV[key]', x)
        V_tensor = mat_tensor_mul(ps.PV[key]', x)
        #tensor_transpose_tensor_mul!(QK_tensor, Q_tensor, K_tensor)
        QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)
        #tensor_tensor_mul!(single_head_output, V_tensor, Lux.softmax(QK_tensor))
        single_head_output = tensor_tensor_mul(V_tensor, Lux.softmax(QK_tensor))
        output = vcat(output, single_head_output) 
        #KernelAbstractions.synchronize(backend)
    end
    x + output, st
end

function Lux.apply(d::MultiHeadAttention{Stiefel, Retraction, false}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {Stiefel, Retraction, T} 
    Dₕ = d.dim ÷ d.n_heads
    dim, input_length, number_data = size(x)
    @assert dim == d.dimoutput
    
    #backend = KernelAbstractions.get_backend(x)

    #output = KernelAbstractions.zeros(backend, T, 0, input_length, number_data)
    output = similar(x, 0, input_length, number_data)

    #Q_tensor = KernelAbstractions.zeros(backend, T, Dₕ, input_length, number_data)
    Q_tensor = similar(x, Dₕ, input_length, number_data)
    #K_tensor = KernelAbstractions.zeros(backend, T, Dₕ, input_length, number_data)
    K_tensor = similar(x, Dₕ, input_length, number_data)
    #V_tensor = KernelAbstractions.zeros(backend, T, Dₕ, input_length, number_data)
    V_tensor = similar(x, Dₕ, input_length, number_data)
    #QK_tensor = KernelAbstractions.zeros(backend, T, input_length, input_length, number_data)
    QK_tensor = similar(x, input_length, input_length, number_data)
    #single_head_output = KernelAbstractions.zeros(backend, T, Dₕ, input_length, number_data)
    single_head_output = similar(x, Dₕ, input_length, number_data)

    for i in 1:d.n_heads 
        key = Symbol("head_"*string(i))
        #use tensor_mat_mul for this computation
        #mat_tensor_mul!(Q_tensor, ps.PQ[key]', x)
        Q_tensor = mat_tensor_mul(ps.PQ[key]', x)
        #mat_tensor_mul!(K_tensor, ps.PK[key]', x)
        K_tensor = mat_tensor_mul(ps.PK[key]', x)
        #mat_tensor_mul!(V_tensor, ps.PV[key]', x)
        V_tensor = mat_tensor_mul(ps.PV[key]', x)
        #tensor_transpose_tensor_mul!(QK_tensor, Q_tensor, K_tensor)
        QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)
        #tensor_tensor_mul!(single_head_output, V_tensor, Lux.softmax(QK_tensor))
        single_head_output = tensor_tensor_mul(V_tensor, Lux.softmax(QK_tensor))
        output = vcat(output, single_head_output) 
        #KernelAbstractions.synchronize(backend)
    end
    output, st
end

import ChainRules
function ChainRules._adjoint_mat_pullback(y::AbstractArray{T, 3}, proj) where T 
    (NoTangent(), proj(tensor_transpose(y)))
end

function mat_tensor_mul(Y::AT, x::AbstractArray{T, 3}) where {T, BT <: AbstractArray{T}, ST <: StiefelManifold{T, BT}, AT <: Adjoint{T, ST}}
    mat_tensor_mul(Y.parent.A', x)
end

function mat_tensor_mul(Y::StiefelManifold, x::AbstractArray{T, 3})
    mat_tensor_mul(Y.A, x)
end