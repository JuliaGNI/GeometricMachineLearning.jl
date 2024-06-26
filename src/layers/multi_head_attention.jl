"""
MultiHeadAttention (MHA) serves as a preprocessing step in the transformer. It reweights the input vectors bases on correlations within those data. 

### Constructor 
Takes input arguments: 
- `dim::Int`: The system dimension 
- `n_heads::Int`: The number of heads. 
- `Stiefel::Bool=true` (keyword argument): whether the weights should be put on the Stiefel manifold. 
- `retraction::AbstractRetraction` (keyword argument): what kind of retraction should be used. By default this is the geodesic retraction. 
- `add_connection::Bool=true` (keyword argument): determines if the input should be added to the output for the final result. 
"""
struct MultiHeadAttention{M, N, Stiefel, retraction, add_connection} <: LayerWithOptionalManifold{M, N, Stiefel, retraction}
    n_heads::Int
end

default_retr = Geodesic()
function MultiHeadAttention(dim::Int, n_heads::Int; Stiefel::Bool=false, retraction::AbstractRetraction=default_retr, add_connection::Bool=true)
    @assert dim % n_heads == 0
    MultiHeadAttention{dim, dim, Stiefel, typeof(retraction), add_connection}(n_heads)
end

function parameterlength(::MultiHeadAttention{M, M, false}) where M
    3*M^2
end

function parameterlength(d::MultiHeadAttention{M, M, true}) where M
    Int(3*M^2 - 3*M*(M + d.n_heads)/(2*d.n_heads))
end

function initialparameters(d::MultiHeadAttention{M, M, false}, backend::KernelAbstractions.Backend, T::Type; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    # number of "hidden" dimension (dimension of projection) 
    Dₕ = M ÷ d.n_heads
    # projections for queries, keys and values.
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


function initialparameters(d::MultiHeadAttention{M, M, true}, backend::KernelAbstractions.Backend, T::Type; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
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

@doc raw"""
Applies MHA to an abstract matrix. This is the same independent of whether the input is added to the output or not. 
"""
function compute_output_of_mha(d::MultiHeadAttention{M, M}, x::AbstractMatrix{T}, ps::NamedTuple) where {M, T}
    dim, input_length = size(x)
    @assert dim == M

    output = typeof(x)(zeros(T, 0, input_length))
    for i in 1:d.n_heads
        key = Symbol("head_"*string(i))
        output = vcat(output, ps.PV[key]' * x * softmax((ps.PQ[key]' * x)' * (ps.PK[key]' * x) / T(sqrt(dim))))
    end
    output
end


function compute_output_of_mha(d::MultiHeadAttention{M, M}, x::AbstractArray{T, 3}, ps::NamedTuple) where {M, T}
    Dₕ = M ÷ d.n_heads
    dim, input_length, number_data = size(x)
    @assert dim == M
    
    # initialize the output
    output = similar(x, 0, input_length, number_data)

    # this is the result of a single head attention block
    single_head_output = similar(x, Dₕ, input_length, number_data)

    for i in 1:d.n_heads 
        key = Symbol("head_"*string(i))
        Q_tensor = mat_tensor_mul(ps.PQ[key]', x)
        K_tensor = mat_tensor_mul(ps.PK[key]', x)
        V_tensor = mat_tensor_mul(ps.PV[key]', x)
        QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)

        single_head_output = tensor_tensor_mul(V_tensor, softmax(QK_tensor/T(sqrt(dim))))
        output = vcat(output, single_head_output) 
    end
    output
end

function (d::MultiHeadAttention{M, M, Stiefel, Retraction, true})(x::AbstractArray, ps::NamedTuple) where {M, Stiefel, Retraction} 
    x + compute_output_of_mha(d, x, ps)
end

function (d::MultiHeadAttention{M, M, Stiefel, Retraction, false})(x::AbstractArray, ps::NamedTuple) where {M, Stiefel, Retraction} 
    compute_output_of_mha(d, x, ps)
end

import ChainRules
# type pyracy! 
function ChainRules._adjoint_mat_pullback(y::AbstractArray{T, 3}, proj) where T 
    (NoTangent(), proj(tensor_transpose(y)))
end

"""
Extend `mat_tensor_mul` to a multiplication by the adjoint of an element of `StiefelManifold`. 
"""
function mat_tensor_mul(Y::AT, x::AbstractArray{T, 3}) where {T, BT <: AbstractArray{T}, ST <: StiefelManifold{T, BT}, AT <: Adjoint{T, ST}}
    mat_tensor_mul(Y.parent.A', x)
end

"""
Extend `mat_tensor_mul` to a multiplication by an element of `StiefelManifold`. 
"""
function mat_tensor_mul(Y::StiefelManifold, x::AbstractArray{T, 3}) where T 
    mat_tensor_mul(Y.A, x)
end