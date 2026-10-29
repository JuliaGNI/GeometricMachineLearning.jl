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
    CanonicalSortLayer{N, M, Axis}()
    CanonicalSortLayer(M::Integer, d::Integer, axis::Integer = 1)

Make an instance of a cotangent-lifted canonical Weyl-chamber particle sorting layer for `M` particles in ``d`` spatial dimensions, sorting along coordinate row `Axis` (`1 <= Axis <= d`, default `1`).

The first type parameter is the phase-space dimension `N = 2Md`, which `input_dimension` and `output_dimension` return.
Given a `(q, p)` `NamedTuple` with `q, p` of shape `(d, M)` (single snapshot) or `(d, M, B)` (batch of ``B`` snapshots), `CanonicalSortLayer{N, M, Axis}()` sorts the `M` particles along coordinate row `Axis` of `q` using a stable merge sort (`MergeSort`), applies the **identical** permutation ``\sigma_b \in S_M`` to both `q` and `p` in each batch sample ``b``, and returns `(q = q_sorted, p = p_sorted, perm = σ)`.
On a flattened phase-space vector of length ``2Md`` or snapshot matrix of shape ``(2Md, B)``, it reshapes each half to `(d, M)` or `(d, M, B)` and returns the sorted flattened state.
"""
struct CanonicalSortLayer{N, M, Axis} <: AbstractExplicitLayer{N, N}
    function CanonicalSortLayer{N, M, Axis}() where {N, M, Axis}
        (N isa Integer && M isa Integer && Axis isa Integer && M >= 1 && Axis >= 1 &&
         N >= 2 * M * Axis && mod(N, 2 * M) == 0) ||
            throw(ArgumentError(
                "CanonicalSortLayer{N, M, Axis} requires M >= 1, Axis >= 1 and N = 2Md with d >= Axis, got N = $N, M = $M, Axis = $Axis."))
        new{Int(N), Int(M), Int(Axis)}()
    end
end

function CanonicalSortLayer(M::Integer, d::Integer, axis::Integer = 1)
    (M >= 1 && d >= 1 && 1 <= axis <= d) ||
        throw(ArgumentError(
            "CanonicalSortLayer requires M >= 1, d >= 1 and 1 <= axis <= d, got M = $M, d = $d, axis = $axis."))
    CanonicalSortLayer{2 * Int(M) * Int(d), Int(M), Int(axis)}()
end

parameterlength(::CanonicalSortLayer) = 0

function initialparameters(
        ::AbstractRNG, ::AbstractNeuralNetworks.Initializer,
        ::CanonicalSortLayer, ::KernelAbstractions.Backend, ::Type{T}) where {T}
    NamedTuple()
end

function _check_sort_dims(
        ::Type{CanonicalSortLayer{N, M, Axis}}, q::AbstractArray, p::AbstractArray) where {
        N, M, Axis}
    d = N ÷ (2 * M)
    (ndims(q) in (2, 3) && size(q) == size(p) && size(q, 1) == d && size(q, 2) == M) ||
        throw(DimensionMismatch(
            "CanonicalSortLayer{$N, $M, $Axis} expects q and p of matching size ($d, $M) or ($d, $M, B), got $(size(q)) and $(size(p))."))
    nothing
end

function _check_first_dim(name::String, N::Int, N2::Int)
    N2 == N ||
        throw(DimensionMismatch("$name expects first dimension $N, got $N2."))
    nothing
end

function (::CanonicalSortLayer{N, M, Axis})(
        qp::NamedTuple{(:q, :p)}, ::NamedTuple = NamedTuple()) where {N, M, Axis}
    _check_sort_dims(CanonicalSortLayer{N, M, Axis}, qp.q, qp.p)
    if ndims(qp.q) == 2
        σ = _sort_perm_single(qp.q, Axis)
        (q = qp.q[:, σ], p = qp.p[:, σ], perm = σ)
    else
        σ = _sort_perm_batch(qp.q, Axis)
        (q = _permute_particles_batch(qp.q, σ),
            p = _permute_particles_batch(qp.p, σ), perm = σ)
    end
end

function (l::CanonicalSortLayer{N, M, Axis})(
        z::AbstractVector, ps::NamedTuple = NamedTuple()) where {N, M, Axis}
    _check_first_dim("CanonicalSortLayer{$N, $M, $Axis}", N, length(z))
    d = N ÷ (2 * M)
    Nq = M * d
    q = reshape(@view(z[1:Nq]), d, M)
    p = reshape(@view(z[(Nq + 1):N]), d, M)
    out = l((q = q, p = p), ps)
    vcat(vec(out.q), vec(out.p))
end

function (l::CanonicalSortLayer{N, M, Axis})(
        z::AbstractMatrix, ps::NamedTuple = NamedTuple()) where {N, M, Axis}
    _check_first_dim("CanonicalSortLayer{$N, $M, $Axis}", N, size(z, 1))
    B = size(z, 2)
    d = N ÷ (2 * M)
    Nq = M * d
    q = reshape(@view(z[1:Nq, :]), d, M, B)
    p = reshape(@view(z[(Nq + 1):N, :]), d, M, B)
    out = l((q = q, p = p), ps)
    vcat(reshape(out.q, Nq, B), reshape(out.p, Nq, B))
end

@doc raw"""
    InvertCanonicalSortLayer{N, M}()
    InvertCanonicalSortLayer(M::Integer, d::Integer)

Make an instance of the analytic inverse of [`CanonicalSortLayer`](@ref) for `M` particles in ``d`` spatial dimensions; the first type parameter is the phase-space dimension `N = 2Md`.

Given `(q = q̂, p = p̂, perm = σ)`, `InvertCanonicalSortLayer{N, M}()` applies ``\sigma_b^{-1}`` (`invperm(σ_b)`) to both `q̂` and `p̂` in each batch sample ``b`` and returns `(q = q_unsorted, p = p_unsorted)`. When called on `(q, p)` or a flattened array without a stored `perm`, it acts as the identity on the canonical Weyl chamber.
"""
struct InvertCanonicalSortLayer{N, M} <: AbstractExplicitLayer{N, N}
    function InvertCanonicalSortLayer{N, M}() where {N, M}
        (N isa Integer && M isa Integer && M >= 1 && N >= 2 * M && mod(N, 2 * M) == 0) ||
            throw(ArgumentError(
                "InvertCanonicalSortLayer{N, M} requires M >= 1 and N = 2Md with d >= 1, got N = $N, M = $M."))
        new{Int(N), Int(M)}()
    end
end

function InvertCanonicalSortLayer(M::Integer, d::Integer)
    (M >= 1 && d >= 1) ||
        throw(ArgumentError(
            "InvertCanonicalSortLayer requires M >= 1 and d >= 1, got M = $M, d = $d."))
    InvertCanonicalSortLayer{2 * Int(M) * Int(d), Int(M)}()
end

parameterlength(::InvertCanonicalSortLayer) = 0

function initialparameters(
        ::AbstractRNG, ::AbstractNeuralNetworks.Initializer,
        ::InvertCanonicalSortLayer, ::KernelAbstractions.Backend, ::Type{T}) where {T}
    NamedTuple()
end

function _check_inv_sort_dims(
        ::Type{InvertCanonicalSortLayer{N, M}}, q::AbstractArray, p::AbstractArray) where {
        N, M}
    d = N ÷ (2 * M)
    (ndims(q) in (2, 3) && size(q) == size(p) && size(q, 1) == d && size(q, 2) == M) ||
        throw(DimensionMismatch(
            "InvertCanonicalSortLayer{$N, $M} expects q and p of matching size ($d, $M) or ($d, $M, B), got $(size(q)) and $(size(p))."))
    nothing
end

function (::InvertCanonicalSortLayer{N, M})(
        qpp::NamedTuple{(:q, :p, :perm)}, ::NamedTuple = NamedTuple()) where {N, M}
    _check_inv_sort_dims(InvertCanonicalSortLayer{N, M}, qpp.q, qpp.p)
    if ndims(qpp.q) == 2
        (ndims(qpp.perm) == 1 && length(qpp.perm) == M) ||
            throw(DimensionMismatch(
                "InvertCanonicalSortLayer{$N, $M} expects perm of size ($M,), got $(size(qpp.perm))."))
        inv_σ = ChainRulesCore.ignore_derivatives(() -> invperm(qpp.perm))
        (q = qpp.q[:, inv_σ], p = qpp.p[:, inv_σ])
    else
        B = size(qpp.q, 3)
        (ndims(qpp.perm) == 2 && size(qpp.perm) == (M, B)) ||
            throw(DimensionMismatch(
                "InvertCanonicalSortLayer{$N, $M} expects perm of size ($M, $B), got $(size(qpp.perm))."))
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

function (::InvertCanonicalSortLayer{N, M})(
        qp::NamedTuple{(:q, :p)}, ::NamedTuple = NamedTuple()) where {N, M}
    _check_inv_sort_dims(InvertCanonicalSortLayer{N, M}, qp.q, qp.p)
    qp
end

function (::InvertCanonicalSortLayer{N, M})(
        z::AbstractVecOrMat, ::NamedTuple = NamedTuple()) where {N, M}
    _check_first_dim("InvertCanonicalSortLayer{$N, $M}", N, size(z, 1))
    z
end

function inverse_layer(::CanonicalSortLayer{N, M, Axis}) where {N, M, Axis}
    InvertCanonicalSortLayer{N, M}()
end
inverse_layer(::InvertCanonicalSortLayer{N, M}) where {N, M} = CanonicalSortLayer{N, M, 1}()
function inverse_layer(::Type{CanonicalSortLayer{N, M, Axis}}) where {N, M, Axis}
    InvertCanonicalSortLayer{N, M}
end
function inverse_layer(::Type{InvertCanonicalSortLayer{N, M}}) where {N, M}
    CanonicalSortLayer{N, M, 1}
end
inverse_layer(::Type{CanonicalSortLayer}) = InvertCanonicalSortLayer
inverse_layer(::Type{InvertCanonicalSortLayer}) = CanonicalSortLayer
