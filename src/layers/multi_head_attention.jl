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

# One projection matrix of a head: dense, or on the Stiefel manifold.
function _head_projection(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::MultiHeadAttention{M, M, false}, Dₕ::Int, backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, T}
    weight = KernelAbstractions.allocate(backend, T, M, Dₕ)
    initializer(rng, weight)
    weight
end

function _head_projection(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::MultiHeadAttention{M, M, true}, Dₕ::Int, backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, T}
    initialparameters(rng, initializer, StiefelLayer{M, Dₕ}(), backend, T).weight
end

# The keys `head_1`, `head_2`, … are built here, once. The draws go head by head, and within a head
# in the order query, key, value.
function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        d::MultiHeadAttention{M, M}, backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, T}
    # the dimension of the projection of one head
    Dₕ = M ÷ d.n_heads
    heads = ntuple(
        _ -> ntuple(_ -> _head_projection(rng, initializer, d, Dₕ, backend, T), 3), d.n_heads)
    keys = ntuple(i -> Symbol("head_", i), d.n_heads)
    (PQ = NamedTuple{keys}(map(h -> h[1], heads)),
        PK = NamedTuple{keys}(map(h -> h[2], heads)),
        PV = NamedTuple{keys}(map(h -> h[3], heads)))
end

# The forward pass reads the heads off the parameters by position, never by a key built at run time.
# The result is a tuple, so the one `vcat` over it infers concretely; a preallocated output cannot
# be written into instead, because Zygote does not differentiate `setindex!`.
function compute_output_of_mha(d::MultiHeadAttention{M, M}, x::AbstractMatrix{T}, ps::NamedTuple) where {
        M, T}
    dim = size(x, 1)
    @assert dim == M

    head_outputs = map(values(ps.PQ), values(ps.PK), values(ps.PV)) do PQ, PK, PV
        _mul(PV', x) * d.activation(_mul(PQ', x)' * _mul(PK', x) / T(sqrt(dim)))
    end

    vcat(head_outputs...)
end

# Apply `d` to `x`, the same way whether or not the input is added to the output.
function compute_output_of_mha(
        d::MultiHeadAttention{M, M}, x::AbstractArray{
            T, 3}, ps::NamedTuple) where {M, T}
    dim = size(x, 1)
    @assert dim == M

    # the result of a single head attention block, one per head
    head_outputs = map(values(ps.PQ), values(ps.PK), values(ps.PV)) do PQ, PK, PV
        Q_tensor = _mul(PQ', x)
        K_tensor = _mul(PK', x)
        V_tensor = _mul(PV', x)
        QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)

        tensor_tensor_mul(V_tensor, d.activation(QK_tensor/T(sqrt(dim))))
    end

    vcat(head_outputs...)
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
