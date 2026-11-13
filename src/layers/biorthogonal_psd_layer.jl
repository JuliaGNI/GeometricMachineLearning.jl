@doc raw"""
    BiorthogonalPSDLayer(input_dim, output_dim)

Make an instance of `BiorthogonalPSDLayer`, a symplectic lift from ``\mathbb{R}^{2n}`` (`input_dim`) to ``\mathbb{R}^{2N}`` (`output_dim`), ``n \leq N``.

The layer is the block-diagonal matrix
```math
A = \begin{bmatrix} \Phi & \mathbb{O} \\ \mathbb{O} & \Psi \end{bmatrix}, \qquad \Psi^T\Phi = \mathbb{I}_n,
```
with a *biorthogonal* pair ``\Phi, \Psi \in \mathbb{R}^{N\times{}n}``. The constraint ``\Psi^T\Phi = \mathbb{I}_n`` is equivalent to ``A^T\mathbb{J}_{2N}A = \mathbb{J}_{2n}``, so ``A`` is symplectic. [`PSDLayer`](@ref) is the case ``\Phi = \Psi \in St(n, N)``.

The pair is the one parameter `weight`, a point of the manifold of biorthogonal pairs [`BiorthogonalManifold`](@extref GeometricOptimizers GeometricOptimizers.BiorthogonalManifold), which stores the ``N\times{}2n`` matrix ``[\Phi\ \ \Psi]``. `parameterlength` is the dimension ``2Nn - n^2`` of that manifold. The initial parameter is the pair ``(U, U)``, with ``U \in St(n, N)`` the weight that `PSDLayer(input_dim, output_dim)` draws with the same random number generator and initializer, so a new layer is the `PSDLayer` with weight ``U``.

Its symplectic inverse ``A^+ = \mathbb{J}_{2n}^TA^T\mathbb{J}_{2N}`` is [`TransposeBiorthogonalPSDLayer`](@ref), returned by [`inverse_layer`](@ref).
"""
struct BiorthogonalPSDLayer{M, N} <: AbstractExplicitLayer{M, N}
    function BiorthogonalPSDLayer{M, N}() where {M, N}
        _validate_biorthogonal_psd_dims(BiorthogonalPSDLayer, M, N, M, N)
        new{Int(M), Int(N)}()
    end
end

@doc raw"""
    TransposeBiorthogonalPSDLayer(input_dim, output_dim)

Make an instance of `TransposeBiorthogonalPSDLayer`, the symplectic inverse of `BiorthogonalPSDLayer(output_dim, input_dim)`, sharing its parameter tuple `ps = (weight = B,)` with `B` the point ``[\Phi\ \ \Psi]`` of the manifold of biorthogonal pairs:
```math
A^+ = \begin{bmatrix} \Psi^T & \mathbb{O} \\ \mathbb{O} & \Phi^T \end{bmatrix}.
```
Since ``\Psi^T\Phi = \mathbb{I}_n``, it is a left inverse of [`BiorthogonalPSDLayer`](@ref) at every point of the manifold.
"""
struct TransposeBiorthogonalPSDLayer{M, N} <: AbstractExplicitLayer{M, N}
    function TransposeBiorthogonalPSDLayer{M, N}() where {M, N}
        _validate_biorthogonal_psd_dims(TransposeBiorthogonalPSDLayer, M, N, N, M)
        new{Int(M), Int(N)}()
    end
end

# `(M, N)` are the layer's input and output dimensions, `latent` and `ambient` those of the lift.
function _validate_biorthogonal_psd_dims(L, M, N, latent, ambient)
    M isa Integer && N isa Integer ||
        throw(ArgumentError("$(nameof(L)) dimensions ($M, $N) must be integers."))
    (M > 0 && N > 0 && iseven(M) && iseven(N)) ||
        throw(ArgumentError("$(nameof(L)) dimensions ($M, $N) must be positive even integers."))
    latent <= ambient ||
        throw(ArgumentError("$(nameof(L)) dimensions ($M, $N): the latent dimension $latent must not exceed the ambient dimension $ambient."))
    nothing
end

BiorthogonalPSDLayer(M::Integer, N::Integer) = BiorthogonalPSDLayer{Int(M), Int(N)}()
TransposeBiorthogonalPSDLayer(M::Integer, N::Integer) = TransposeBiorthogonalPSDLayer{Int(M), Int(N)}()

inverse_layer(::BiorthogonalPSDLayer{M, N}) where {M, N} = TransposeBiorthogonalPSDLayer{N, M}()
inverse_layer(::TransposeBiorthogonalPSDLayer{M, N}) where {M, N} = BiorthogonalPSDLayer{N, M}()
inverse_layer(::Type{BiorthogonalPSDLayer{M, N}}) where {M, N} = TransposeBiorthogonalPSDLayer{N, M}
inverse_layer(::Type{TransposeBiorthogonalPSDLayer{M, N}}) where {M, N} = BiorthogonalPSDLayer{N, M}
inverse_layer(::Type{BiorthogonalPSDLayer}) = TransposeBiorthogonalPSDLayer
inverse_layer(::Type{TransposeBiorthogonalPSDLayer}) = BiorthogonalPSDLayer

# The dimension of the manifold of biorthogonal pairs, `2Nn - n²`, as `PSDLayer` counts the dimension
# of the Stiefel manifold, not the `2Nn` stored entries of `[Φ Ψ]`.
function parameterlength(::BiorthogonalPSDLayer{M, N}) where {M, N}
    n, N2 = M ÷ 2, N ÷ 2
    2 * N2 * n - n^2
end

function parameterlength(::TransposeBiorthogonalPSDLayer{M, N}) where {M, N}
    parameterlength(BiorthogonalPSDLayer{N, M}())
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::BiorthogonalPSDLayer{M, N}, backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, N, T}
    U = parent(initialparameters(rng, initializer, PSDLayer{M, N}(), backend, T).weight)
    (weight = BiorthogonalManifold(hcat(U, U)),)
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::TransposeBiorthogonalPSDLayer{M, N}, backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, N, T}
    initialparameters(rng, initializer, BiorthogonalPSDLayer{N, M}(), backend, T)
end

# `(Φ, Ψ)`, the two halves of the stored matrix `[Φ Ψ]`, as matrices on the point's backend.
function _biorthogonal_pair(B::BiorthogonalManifold)
    A = parent(B)
    n = size(A, 2) ÷ 2
    A[:, 1:n], A[:, (n + 1):(2n)]
end

# The cotangent of the point is the dense `[Φ̄ Ψ̄]`, the Euclidean gradient with respect to the stored
# matrix that `rgrad` projects, as the weight of a `PSDLayer` receives a dense matrix. Without this
# rule Zygote reads `parent` as a field access and returns a structural tangent.
function ChainRulesCore.rrule(::typeof(_biorthogonal_pair), B::BiorthogonalManifold)
    Φ, Ψ = _biorthogonal_pair(B)
    function _biorthogonal_pair_pullback(c̄)
        c̄ = unthunk(c̄)
        c̄ isa AbstractZero && return NoTangent(), c̄
        half(x̄, X) = (x̄ = _seed(eltype(X), x̄); x̄ isa AbstractZero ? zero(X) : x̄)
        NoTangent(), hcat(half(c̄[1], Φ), half(c̄[2], Ψ))
    end
    (Φ, Ψ), _biorthogonal_pair_pullback
end

function (::BiorthogonalPSDLayer{M, N})(qp::NamedTuple{(:q, :p), Tuple{AT1, AT2}},
        ps::NamedTuple) where {M, N, AT1 <: AbstractArray, AT2 <: AbstractArray}
    size(qp.q, 1) == size(qp.p, 1) == M ÷ 2 || error("Dimension mismatch.")
    Φ, Ψ = _biorthogonal_pair(ps.weight)
    (q = _mul(Φ, qp.q), p = _mul(Ψ, qp.p))
end

function (::TransposeBiorthogonalPSDLayer{M, N})(qp::NamedTuple{(:q, :p), Tuple{AT1, AT2}},
        ps::NamedTuple) where {M, N, AT1 <: AbstractArray, AT2 <: AbstractArray}
    size(qp.q, 1) == size(qp.p, 1) == M ÷ 2 || error("Dimension mismatch.")
    Φ, Ψ = _biorthogonal_pair(ps.weight)
    (q = _mul(Ψ', qp.q), p = _mul(Φ', qp.p))
end

function (l::Union{BiorthogonalPSDLayer{M}, TransposeBiorthogonalPSDLayer{M}})(
        x::AbstractArray, ps::NamedTuple) where {M}
    size(x, 1) == M || error("Dimension mismatch.")
    _vcat(l(assign_q_and_p(x, M ÷ 2), ps))
end
