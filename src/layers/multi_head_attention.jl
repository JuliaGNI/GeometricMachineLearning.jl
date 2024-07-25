@doc raw"""
    MultiHeadAttention(dim, n_heads)

Make a `MultiHeadAttention` layer with `n_heads` for a system of dimension `dim`. 

Note that the `dim` has to be divisible by `n_heads`.

MultiHeadAttention (MHA) serves as a preprocessing step in the transformer. 

It reweights the input vectors bases on correlations within those data.

This is used for the neural networks [`StandardTransformerIntegrator`](@ref) and [`ClassificationTransformer`](@ref).

# Arguments

The optional keyword arguments to `MultiHeadAttention` are:
- `Stiefel::Bool`
- `add_connection::Bool`

`Stiefel` indicates whether weights are put on ``St(\mathrm{dim}, \mathrm{dim}\div\mathrm{n_heads})``.

`add_connection` indicates whether the input is again added to the output.
"""
struct MultiHeadAttention{M, N, Stiefel, add_connection} <: AbstractExplicitLayer{M, N}
    n_heads::Int
end

function MultiHeadAttention(dim::Int, n_heads::Int; Stiefel::Bool=false, add_connection::Bool=true)
    @assert dim % n_heads == 0
    MultiHeadAttention{dim, dim, Stiefel, add_connection}(n_heads)
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

@doc raw"""
    compute_output_of_mha(d::MultiHeadAttention, x, ps)

Apply [`MultiHeadAttention`](@ref) layer `d` to `x`. 

This is the same, independent of whether the input is added to the output or not. 
"""
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

function (d::MultiHeadAttention{M, M, Stiefel, true})(x::AbstractArray, ps::NamedTuple) where {M, Stiefel} 
    x + compute_output_of_mha(d, x, ps)
end

function (d::MultiHeadAttention{M, M, Stiefel, false})(x::AbstractArray, ps::NamedTuple) where {M, Stiefel} 
    compute_output_of_mha(d, x, ps)
end

import ChainRules
# type pyracy! 
function ChainRules._adjoint_mat_pullback(y::AbstractArray{T, 3}, proj) where T 
    (NoTangent(), proj(tensor_transpose(y)))
end

function mat_tensor_mul(Y::AT, x::AbstractArray{T, 3}) where {T, BT <: AbstractArray{T}, ST <: StiefelManifold{T, BT}, AT <: Adjoint{T, ST}}
    mat_tensor_mul(Y.parent.A', x)
end

@doc raw"""
    mat_tensor_mul(Y::StiefelManifold, x::AbstractArray{<:Number, 3})

Multiply `Y` with all matrices stored in `x` (parallelize over the third axis).
"""
function mat_tensor_mul(Y::StiefelManifold, x::AbstractArray{T, 3}) where T 
    mat_tensor_mul(Y.A, x)
end