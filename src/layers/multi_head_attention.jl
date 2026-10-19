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

# One projection of one head, `M × Dₕ`: a free weight, or one with orthonormal columns when the
# layer's projections live on the Stiefel manifold.
function _projection_weight(
        rng::AbstractRNG, initializer, ::Val{false}, ::Val{M}, ::Val{Dₕ},
        backend::KernelAbstractions.Backend, ::Type{T}) where {M, Dₕ, T}
    W = KernelAbstractions.allocate(backend, T, M, Dₕ)
    initializer(rng, W)
    W
end

function _projection_weight(
        rng::AbstractRNG, initializer, ::Val{true}, ::Val{M}, ::Val{Dₕ},
        backend::KernelAbstractions.Backend, ::Type{T}) where {M, Dₕ, T}
    only(values(initialparameters(rng, initializer, StiefelLayer{M, Dₕ}(), backend, T)))
end

# The head keys are built here, once, and the forward pass reads them off these `NamedTuple`s. The
# `Stiefel` flag is a type parameter, so the projection's two forms are two methods and not a
# branch. The draw order is one projection per head for all queries, then all keys, then all
# values.
function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        d::MultiHeadAttention{M, M, Stiefel},
        backend::KernelAbstractions.Backend, ::Type{T}) where {M, Stiefel, T}
    Dₕ = M ÷ d.n_heads
    keys = ntuple(head -> Symbol("head_", head), d.n_heads)
    heads() = NamedTuple{keys}(ntuple(
        _ -> _projection_weight(rng, initializer, Val(Stiefel),
            Val(M), Val(Dₕ), backend, T),
        d.n_heads))
    (PQ = heads(), PK = heads(), PV = heads())
end

# One head's output: its value projection applied to the attention of its query and key
# projections. `PQ`, `PK` and `PV` are the three weights of that head, taken off the parameter set,
# so nothing here builds a key.
function head_output(d, PQ, PK, PV, x::AbstractMatrix{T}) where {T}
    _mul(PV', x) * d.activation(_mul(PQ', x)' * _mul(PK', x) / T(sqrt(size(x, 1))))
end

function compute_output_of_mha(d::MultiHeadAttention{M, M}, x::AbstractMatrix{T}, ps::NamedTuple) where {
        M, T}
    @assert size(x, 1) == M

    # `map` over the three parameter `NamedTuple`s pairs the heads' weights without naming a key,
    # and its arguments' lengths are in their types, so the tuple of head outputs is known here.
    head_outputs = map((PQ, PK, PV) -> head_output(d, PQ, PK, PV, x), ps.PQ, ps.PK, ps.PV)

    # One `vcat` over all heads, not one per head. The head outputs are matrices, so `reduce` takes
    # Base's linear path and infers concretely without an assertion. The tensor method below cannot
    # use it and says why. `values` and not the `NamedTuple` itself: `reduce` over one maps into
    # another and allocates that, while the value tuple reduces without an allocation of its own.
    reduce(vcat, values(head_outputs))
end

# The same head output on a batch of sequences.
function head_output(d, PQ, PK, PV, x::AbstractArray{T, 3}) where {T}
    Q_tensor = _mul(PQ', x)
    K_tensor = _mul(PK', x)
    V_tensor = _mul(PV', x)
    QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)

    tensor_tensor_mul(V_tensor, d.activation(QK_tensor/T(sqrt(size(x, 1)))))
end

# Apply `d` to `x`, the same way whether or not the input is added to the output.
function compute_output_of_mha(
        d::MultiHeadAttention{M, M}, x::AbstractArray{
            T, 3}, ps::NamedTuple) where {M, T}
    @assert size(x, 1) == M

    head_outputs = map((PQ, PK, PV) -> head_output(d, PQ, PK, PV, x), ps.PQ, ps.PK, ps.PV)

    # One variadic `vcat` over all heads, not one per head. Two things constrain this: a
    # preallocated output cannot be written into, because Zygote does not differentiate
    # `setindex!`; and `reduce(vcat, …)` takes its linear path only for vectors and matrices, so on
    # a 3-tensor it folds pairwise and stays quadratic. `values` gives a tuple, whose length is in
    # its type, so the splat's argument count is known and `vcat` infers concretely.
    vcat(values(head_outputs)...)
end

function (d::MultiHeadAttention{M, M, Stiefel, true})(x::AbstractArray, ps::NamedTuple) where {
        M, Stiefel}
    x + compute_output_of_mha(d, x, ps)
end

function (d::MultiHeadAttention{M, M, Stiefel, false})(x::AbstractArray, ps::NamedTuple) where {
        M, Stiefel}
    compute_output_of_mha(d, x, ps)
end

function (d::MultiHeadAttention)(z::QPT, ps::NamedTuple)
    N2 = size(z.q, 1)
    output = d(vcat(z.q, z.p), ps)
    assign_q_and_p(output, N2)
end
