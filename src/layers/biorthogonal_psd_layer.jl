@doc raw"""
    BiorthogonalPSDLayer(input_dim, output_dim)

Make an instance of `BiorthogonalPSDLayer`, a symplectic lift from ``\mathbb{R}^{2n}`` (`input_dim`) to ``\mathbb{R}^{2N}`` (`output_dim`), ``n \leq N``.

The layer is the block-diagonal matrix
```math
A = \begin{bmatrix} \Phi & \mathbb{O} \\ \mathbb{O} & \Psi \end{bmatrix}, \qquad \Psi^T\Phi = \mathbb{I}_n,
```
with a *biorthogonal* pair ``\Phi, \Psi \in \mathbb{R}^{N\times{}n}``. The constraint ``\Psi^T\Phi = \mathbb{I}_n`` is equivalent to ``A^T\mathbb{J}_{2N}A = \mathbb{J}_{2n}``, so ``A`` is symplectic. [`PSDLayer`](@ref) is the case ``\Phi = \Psi \in St(n, N)``.

The pair is parametrized by ``U \in St(n, N)`` (`weight`), ``R \in \mathbb{R}^{n\times{}n}`` (`factor`) and ``W \in \mathbb{R}^{N\times{}n}`` (`complement`):
```math
\Phi = UR, \qquad \Psi = UR^{-T} + (\mathbb{I}_N - UU^T)W,
```
the parametrization of the manifold of biorthogonal pairs derived in [The bundle over the first factor](@extref GeometricOptimizers The-bundle-over-the-first-factor). Only ``U`` is constrained; ``R`` and ``W`` are trained as Euclidean weights. `parameterlength` is the dimension ``2Nn - n^2`` of that manifold, [`BiorthogonalManifold`](@extref GeometricOptimizers GeometricOptimizers.BiorthogonalManifold). The initial parameters are ``R = \mathbb{I}_n`` and ``W = \mathbb{O}``, so a new layer is the `PSDLayer` with weight ``U``.

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

Make an instance of `TransposeBiorthogonalPSDLayer`, the symplectic inverse of `BiorthogonalPSDLayer(output_dim, input_dim)`, sharing its parameter tuple `ps = (weight = U, factor = R, complement = W)`:
```math
A^+ = \begin{bmatrix} \Psi^T & \mathbb{O} \\ \mathbb{O} & \Phi^T \end{bmatrix}.
```
Since ``\Psi^T\Phi = \mathbb{I}_n``, it is a left inverse of [`BiorthogonalPSDLayer`](@ref) for every value of the parameters.
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
        throw(ArgumentError("$L dimensions ($M, $N) must be integers."))
    (M > 0 && N > 0 && iseven(M) && iseven(N)) ||
        throw(ArgumentError("$L dimensions ($M, $N) must be positive even integers."))
    latent <= ambient ||
        throw(ArgumentError("$L dimensions ($M, $N): the latent dimension $latent must not exceed the ambient dimension $ambient."))
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
# of the Stiefel manifold. The parameters `(U, R, W)` have more entries: `R` and `UR` give the same `Φ`
# up to an orthogonal matrix, and the part `UUᵀW` of `W` does not enter `Ψ`.
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
    n, N2 = M ÷ 2, N ÷ 2
    factor = KernelAbstractions.zeros(backend, T, n, n)
    factor[diagind(factor)] .= one(T)
    (weight = initialparameters(rng, initializer, PSDLayer{M, N}(), backend, T).weight,
        factor = factor,
        complement = KernelAbstractions.zeros(backend, T, N2, n))
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::TransposeBiorthogonalPSDLayer{M, N}, backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, N, T}
    initialparameters(rng, initializer, BiorthogonalPSDLayer{N, M}(), backend, T)
end

# Φx = U(Rx)
_biorthogonal_phi(ps::NamedTuple, x) = _mul(ps.weight, _mul(ps.factor, x))
# Ψx = U(R⁻ᵀx) + Wx - U(UᵀWx)
function _biorthogonal_psi(ps::NamedTuple, x)
    Wx = _mul(ps.complement, x)
    _mul(ps.weight, _mul(inv(ps.factor)', x)) + Wx - _mul(ps.weight, _mul(ps.weight', Wx))
end
# Φᵀy = Rᵀ(Uᵀy)
_biorthogonal_phi_t(ps::NamedTuple, y) = _mul(ps.factor', _mul(ps.weight', y))
# Ψᵀy = R⁻¹(Uᵀy) + Wᵀy - Wᵀ(U(Uᵀy))
function _biorthogonal_psi_t(ps::NamedTuple, y)
    Uty = _mul(ps.weight', y)
    _mul(inv(ps.factor), Uty) + _mul(ps.complement', y) -
    _mul(ps.complement', _mul(ps.weight, Uty))
end

function (::BiorthogonalPSDLayer{M, N})(qp::NamedTuple{(:q, :p), Tuple{AT1, AT2}},
        ps::NamedTuple) where {M, N, AT1 <: AbstractArray, AT2 <: AbstractArray}
    size(qp.q, 1) == size(qp.p, 1) == M ÷ 2 || error("Dimension mismatch.")
    (q = _biorthogonal_phi(ps, qp.q), p = _biorthogonal_psi(ps, qp.p))
end

function (::TransposeBiorthogonalPSDLayer{M, N})(qp::NamedTuple{(:q, :p), Tuple{AT1, AT2}},
        ps::NamedTuple) where {M, N, AT1 <: AbstractArray, AT2 <: AbstractArray}
    size(qp.q, 1) == size(qp.p, 1) == M ÷ 2 || error("Dimension mismatch.")
    (q = _biorthogonal_psi_t(ps, qp.q), p = _biorthogonal_phi_t(ps, qp.p))
end

function (l::Union{BiorthogonalPSDLayer{M}, TransposeBiorthogonalPSDLayer{M}})(
        x::AbstractArray, ps::NamedTuple) where {M}
    size(x, 1) == M || error("Dimension mismatch.")
    _vcat(l(assign_q_and_p(x, M ÷ 2), ps))
end
