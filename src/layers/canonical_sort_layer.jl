function _sort_perm_single(q::AbstractMatrix, axis::Int)
    ChainRulesCore.ignore_derivatives() do
        sortperm(collect(@view(q[axis, :])); alg = MergeSort)
    end
end

function _sort_perm_batch(q::AbstractArray{<:Any, 3}, axis::Int)
    ChainRulesCore.ignore_derivatives() do
        M = size(q, 2)
        B = size(q, 3)
        perms = Matrix{Int}(undef, M, B)
        for b in 1:B
            perms[:, b] .= sortperm(collect(@view(q[axis, :, b])); alg = MergeSort)
        end
        perms
    end
end

function _permute_particles_batch(
        x::AbstractArray{T, 3}, perms::AbstractMatrix{<:Integer}) where {T}
    d, M, B = size(x)
    lin_idx = ChainRulesCore.ignore_derivatives() do
        vec([perms[i, b] + (b - 1) * M for i in 1:M, b in 1:B])
    end
    reshape(reshape(x, d, M * B)[:, lin_idx], d, M, B)
end

@doc raw"""
    CanonicalSortLayer{M, Axis}()
    CanonicalSortLayer(M::Integer, axis::Integer = 1)

Make an instance of a cotangent-lifted canonical Weyl-chamber particle sorting layer for `M` particles sorting along coordinate row `Axis` (default `1`).

Given a `(q, p)` `NamedTuple` with `q, p` of shape `(d, M)` (single snapshot) or `(d, M, B)` (batch of ``B`` snapshots), `CanonicalSortLayer{M, Axis}()` sorts the `M` particles along coordinate row `Axis` of `q` using a stable merge sort (`MergeSort`), applies the **identical** permutation ``\sigma_b \in S_M`` to both `q` and `p` in each batch sample ``b``, and returns `(q = q_sorted, p = p_sorted, perm = σ)`.
On a flattened phase-space vector of length ``2Md`` or snapshot matrix of shape ``(2Md, B)``, it reshapes each half to `(d, M)` or `(d, M, B)` and returns the sorted flattened state.
"""
struct CanonicalSortLayer{M, Axis} <: AbstractExplicitLayer{M, M}
    function CanonicalSortLayer{M, Axis}() where {M, Axis}
        (M isa Integer && Axis isa Integer && M >= 1 && Axis >= 1) ||
            throw(ArgumentError(
                "CanonicalSortLayer requires M >= 1 and Axis >= 1, got M = $M, Axis = $Axis."))
        new{Int(M), Int(Axis)}()
    end
end

CanonicalSortLayer{M}() where {M} = CanonicalSortLayer{M, 1}()
CanonicalSortLayer(M::Integer, axis::Integer = 1) = CanonicalSortLayer{Int(M), Int(axis)}()

parameterlength(::CanonicalSortLayer) = 0

function initialparameters(
        ::AbstractRNG, ::AbstractNeuralNetworks.Initializer,
        ::CanonicalSortLayer, ::KernelAbstractions.Backend, ::Type{T}) where {T}
    NamedTuple()
end

function _check_sort_dims(
        ::Type{CanonicalSortLayer{M, Axis}}, q::AbstractArray, p::AbstractArray) where {
        M, Axis}
    nd = ndims(q)
    (nd in (2, 3) && size(q) == size(p) && size(q, 2) == M && size(q, 1) >= Axis) ||
        throw(DimensionMismatch(
            "CanonicalSortLayer{$M, $Axis} expects q and p of matching size (d, $M) or (d, $M, B) with d >= $Axis, got $(size(q)) and $(size(p))."))
    nothing
end

function (::CanonicalSortLayer{M, Axis})(
        qp::NamedTuple{(:q, :p)}, ::NamedTuple = NamedTuple()) where {M, Axis}
    _check_sort_dims(CanonicalSortLayer{M, Axis}, qp.q, qp.p)
    if ndims(qp.q) == 2
        σ = _sort_perm_single(qp.q, Axis)
        (q = qp.q[:, σ], p = qp.p[:, σ], perm = σ)
    else
        σ = _sort_perm_batch(qp.q, Axis)
        (q = _permute_particles_batch(qp.q, σ),
            p = _permute_particles_batch(qp.p, σ), perm = σ)
    end
end

function (l::CanonicalSortLayer{M, Axis})(
        z::AbstractVector, ps::NamedTuple = NamedTuple()) where {M, Axis}
    N2 = length(z)
    (N2 >= 2 * M * Axis && mod(N2, 2 * M) == 0) ||
        throw(DimensionMismatch(
            "CanonicalSortLayer{$M, $Axis} expects first dimension divisible by $(2 * M) with d >= $Axis, got $N2."))
    d = N2 ÷ (2 * M)
    N = M * d
    q = reshape(@view(z[1:N]), d, M)
    p = reshape(@view(z[(N + 1):N2]), d, M)
    out = l((q = q, p = p), ps)
    vcat(vec(out.q), vec(out.p))
end

function (l::CanonicalSortLayer{M, Axis})(
        z::AbstractMatrix, ps::NamedTuple = NamedTuple()) where {M, Axis}
    N2, B = size(z)
    (N2 >= 2 * M * Axis && mod(N2, 2 * M) == 0) ||
        throw(DimensionMismatch(
            "CanonicalSortLayer{$M, $Axis} expects first dimension divisible by $(2 * M) with d >= $Axis, got $N2."))
    d = N2 ÷ (2 * M)
    N = M * d
    q = reshape(@view(z[1:N, :]), d, M, B)
    p = reshape(@view(z[(N + 1):N2, :]), d, M, B)
    out = l((q = q, p = p), ps)
    vcat(reshape(out.q, N, B), reshape(out.p, N, B))
end

@doc raw"""
    InvertCanonicalSortLayer{M}()
    InvertCanonicalSortLayer(M::Integer)

Make an instance of the analytic inverse of [`CanonicalSortLayer`](@ref) for `M` particles.

Given `(q = q̂, p = p̂, perm = σ)`, `InvertCanonicalSortLayer{M}()` applies ``\sigma_b^{-1}`` (`invperm(σ_b)`) to both `q̂` and `p̂` in each batch sample ``b`` and returns `(q = q_unsorted, p = p_unsorted)`. When called on `(q, p)` or a flattened array without a stored `perm`, it acts as the identity on the canonical Weyl chamber.
"""
struct InvertCanonicalSortLayer{M} <: AbstractExplicitLayer{M, M}
    function InvertCanonicalSortLayer{M}() where {M}
        (M isa Integer && M >= 1) ||
            throw(ArgumentError("InvertCanonicalSortLayer requires M >= 1, got M = $M."))
        new{Int(M)}()
    end
end

InvertCanonicalSortLayer(M::Integer) = InvertCanonicalSortLayer{Int(M)}()

parameterlength(::InvertCanonicalSortLayer) = 0

function initialparameters(
        ::AbstractRNG, ::AbstractNeuralNetworks.Initializer,
        ::InvertCanonicalSortLayer, ::KernelAbstractions.Backend, ::Type{T}) where {T}
    NamedTuple()
end

function _check_inv_sort_dims(
        ::Type{InvertCanonicalSortLayer{M}}, q::AbstractArray, p::AbstractArray) where {M}
    nd = ndims(q)
    (nd in (2, 3) && size(q) == size(p) && size(q, 2) == M) ||
        throw(DimensionMismatch(
            "InvertCanonicalSortLayer{$M} expects q and p of matching size (d, $M) or (d, $M, B), got $(size(q)) and $(size(p))."))
    nothing
end

function (::InvertCanonicalSortLayer{M})(
        qpp::NamedTuple{(:q, :p, :perm)}, ::NamedTuple = NamedTuple()) where {M}
    _check_inv_sort_dims(InvertCanonicalSortLayer{M}, qpp.q, qpp.p)
    if ndims(qpp.q) == 2
        (ndims(qpp.perm) == 1 && length(qpp.perm) == M) ||
            throw(DimensionMismatch(
                "InvertCanonicalSortLayer{$M} expects perm of size ($M,), got $(size(qpp.perm))."))
        inv_σ = ChainRulesCore.ignore_derivatives(() -> invperm(qpp.perm))
        (q = qpp.q[:, inv_σ], p = qpp.p[:, inv_σ])
    else
        B = size(qpp.q, 3)
        (ndims(qpp.perm) == 2 && size(qpp.perm) == (M, B)) ||
            throw(DimensionMismatch(
                "InvertCanonicalSortLayer{$M} expects perm of size ($M, $B), got $(size(qpp.perm))."))
        inv_σ = ChainRulesCore.ignore_derivatives() do
            inv_perms = Matrix{Int}(undef, M, B)
            for b in 1:B
                inv_perms[:, b] .= invperm(@view(qpp.perm[:, b]))
            end
            inv_perms
        end
        (q = _permute_particles_batch(qpp.q, inv_σ),
            p = _permute_particles_batch(qpp.p, inv_σ))
    end
end

function (::InvertCanonicalSortLayer{M})(
        qp::NamedTuple{(:q, :p)}, ::NamedTuple = NamedTuple()) where {M}
    _check_inv_sort_dims(InvertCanonicalSortLayer{M}, qp.q, qp.p)
    qp
end

function (::InvertCanonicalSortLayer{M})(
        z::AbstractVecOrMat, ::NamedTuple = NamedTuple()) where {M}
    N2 = size(z, 1)
    (N2 >= 2 * M && mod(N2, 2 * M) == 0) ||
        throw(DimensionMismatch(
            "InvertCanonicalSortLayer{$M} expects first dimension divisible by $(2 * M), got $N2."))
    z
end

inverse_layer(::CanonicalSortLayer{M, Axis}) where {M, Axis} = InvertCanonicalSortLayer{M}()
inverse_layer(::InvertCanonicalSortLayer{M}) where {M} = CanonicalSortLayer{M, 1}()
function inverse_layer(::Type{CanonicalSortLayer{M, Axis}}) where {M, Axis}
    InvertCanonicalSortLayer{M}
end
inverse_layer(::Type{CanonicalSortLayer{M}}) where {M} = InvertCanonicalSortLayer{M}
inverse_layer(::Type{InvertCanonicalSortLayer{M}}) where {M} = CanonicalSortLayer{M, 1}
inverse_layer(::Type{CanonicalSortLayer}) = InvertCanonicalSortLayer
inverse_layer(::Type{InvertCanonicalSortLayer}) = CanonicalSortLayer
