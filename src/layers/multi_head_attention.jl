@doc raw"""
    MultiHeadAttention(dim, n_heads)

Make a `MultiHeadAttention` layer with `n_heads` for a system of dimension `dim`. 

Note that the `dim` has to be divisible by `n_heads`.

MultiHeadAttention (MHA) serves as a preprocessing step in the transformer. 

It reweights the input vectors bases on correlations within those data.

This is used for the neural networks [`StandardTransformerIntegrator`](@ref) and [`ClassificationTransformer`](@ref).

# Arguments

The optional keyword arguments to `MultiHeadAttention` are:
- `Stiefel::Bool=false`
- `add_connection::Bool=true`
- `activation::AbstractSoftmax=`[`VectorSoftmax`](@ref).

`Stiefel` indicates whether weights are put on the [`StiefelManifold`](@extref GeometricOptimizers GeometricOptimizers.StiefelManifold) ``St(\mathrm{dim}, \mathrm{dim}\div\mathrm{n\_heads})``.

`add_connection` indicates whether the input is again added to the output.
"""
struct MultiHeadAttention{M, N, Stiefel, add_connection, AT <: AbstractSoftmax} <:
       AbstractExplicitLayer{M, N}
    n_heads::Int
    activation::AT
end

function MultiHeadAttention(dim::Int, n_heads::Int; Stiefel::Bool = false,
        add_connection::Bool = true,
        activation::AbstractSoftmax = VectorSoftmax())
    @assert dim % n_heads == 0
    MultiHeadAttention{dim, dim, Stiefel, add_connection, typeof(activation)}(n_heads, activation)
end

function parameterlength(::MultiHeadAttention{M, M, false}) where {M}
    3*M^2
end

function parameterlength(d::MultiHeadAttention{M, M, true}) where {M}
    3*M^2 - (3*M*(M + d.n_heads)) ÷ (2*d.n_heads)
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        d::MultiHeadAttention{M, M, false},
        backend::KernelAbstractions.Backend, T::Type) where {M}
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
            NamedTuple{(key,)}((PQ_weight,))
        )
        PK = merge(PK,
            NamedTuple{(key,)}((PK_weight,))
        )
        PV = merge(PV,
            NamedTuple{(key,)}((PV_weight,))
        )
    end
    (PQ = PQ, PK = PK, PV = PV)
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        d::MultiHeadAttention{M, M, true},
        backend::KernelAbstractions.Backend, ::Type{T}) where {M, T}
    # number of "hidden" dimension (dimension of projection) 
    Dₕ = M ÷ d.n_heads
    # projections for queries, keys and vectors.
    PQ = NamedTuple()
    PK = NamedTuple()
    PV = NamedTuple()

    for head in 1:d.n_heads
        key = Symbol("head_"*string(head))

        PQ = merge(PQ,
            NamedTuple{(key,)}(values(initialparameters(
                rng, initializer, StiefelLayer{M, Dₕ}(), backend, T)))
        )
        PK = merge(PK,
            NamedTuple{(key,)}(values(initialparameters(
                rng, initializer, StiefelLayer{M, Dₕ}(), backend, T)))
        )
        PV = merge(PV,
            NamedTuple{(key,)}(values(initialparameters(
                rng, initializer, StiefelLayer{M, Dₕ}(), backend, T)))
        )
    end
    (PQ = PQ, PK = PK, PV = PV)
end

function compute_output_of_mha(d::MultiHeadAttention{M, M}, x::AbstractMatrix{T}, ps::NamedTuple) where {
        M, T}
    dim = size(x, 1)
    @assert dim == M

    head_outputs = map(1:(d.n_heads)) do i
        key = Symbol("head_", i)
        ps.PV[key]' * x *
        d.activation((ps.PQ[key]' * x)' * (ps.PK[key]' * x) / T(sqrt(dim)))
    end

    # One `vcat` over all heads, not one per head. The head outputs are matrices, so `reduce` takes
    # Base's linear path and infers concretely without an assertion. The tensor method below cannot
    # use it and says why.
    reduce(vcat, head_outputs)
end

# Apply `d` to `x`, the same way whether or not the input is added to the output.
function compute_output_of_mha(
        d::MultiHeadAttention{M, M}, x::AbstractArray{
            T, 3}, ps::NamedTuple) where {M, T}
    dim = size(x, 1)
    @assert dim == M

    # the result of a single head attention block, one per head
    head_outputs = map(1:(d.n_heads)) do i
        key = Symbol("head_", i)
        Q_tensor = mat_tensor_mul(ps.PQ[key]', x)
        K_tensor = mat_tensor_mul(ps.PK[key]', x)
        V_tensor = mat_tensor_mul(ps.PV[key]', x)
        QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)

        tensor_tensor_mul(V_tensor, d.activation(QK_tensor/T(sqrt(dim))))
    end

    # One variadic `vcat` over all heads, not one per head. Two things constrain this: a
    # preallocated output cannot be written into, because Zygote does not differentiate
    # `setindex!`; and `reduce(vcat, …)` takes its linear path only for vectors and matrices, so on
    # a 3-tensor it folds pairwise and stays quadratic. Splatting a vector in turn hides the
    # argument count from inference, which makes the call return `Any`, so the result type is
    # asserted: a concatenation has the type of the pieces it concatenates.
    vcat(head_outputs...)::eltype(head_outputs)
end

function (d::MultiHeadAttention{M, M, Stiefel, true})(x::AbstractArray, ps::NamedTuple) where {
        M, Stiefel}
    x + compute_output_of_mha(d, x, ps)
end

function (d::MultiHeadAttention{M, M, Stiefel, false})(x::AbstractArray, ps::NamedTuple) where {
        M, Stiefel}
    compute_output_of_mha(d, x, ps)
end

function mat_tensor_mul(Y::AT,
        x::AbstractArray{T, 3}) where {T <: Number,
        BT <: AbstractArray{T},
        ST <: StiefelManifold{T, BT},
        AT <: Adjoint{T, ST}}
    mat_tensor_mul(Y.parent.A', x)
end

# Multiply `Y` with every matrix stored in `x`, parallelised over the third axis.
function mat_tensor_mul(Y::StiefelManifold, x::AbstractArray{<:Number, 3})
    mat_tensor_mul(Y.A, x)
end

function (d::MultiHeadAttention)(z::QPT, ps::NamedTuple)
    N2 = size(z.q, 1)
    output = d(vcat(z.q, z.p), ps)
    assign_q_and_p(output, N2)
end
