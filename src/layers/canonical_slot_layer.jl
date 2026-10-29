function _helmert_relative_matrix(::Type{T}, M::Int) where {T}
    H = zeros(T, M, M - 1)
    for k in 1:(M - 1)
        s = inv(sqrt(T(k * (k + 1))))
        for i in 1:k
            H[i, k] = s
        end
        H[k + 1, k] = -T(k) * s
    end
    H
end

function _sort_relative_modes_2d(q_rel::AbstractMatrix, p_rel::AbstractMatrix)
    perm = ChainRulesCore.ignore_derivatives() do
        M_rel = size(q_rel, 2)
        norms = [sum(abs2, @view(q_rel[:, k])) + sum(abs2, @view(p_rel[:, k]))
                 for k in 1:M_rel]
        sortperm(norms; rev = true, alg = MergeSort)
    end
    (q = q_rel[:, perm], p = p_rel[:, perm], perm = perm)
end

function _sort_relative_modes_3d(
        q_rel::AbstractArray{T, 3}, p_rel::AbstractArray{T, 3}) where {T}
    d, M_rel, B = size(q_rel)
    lin_idx = ChainRulesCore.ignore_derivatives() do
        perms = Matrix{Int}(undef, M_rel, B)
        for b in 1:B
            norms = [sum(abs2, @view(q_rel[:, k, b])) + sum(abs2, @view(p_rel[:, k, b]))
                     for k in 1:M_rel]
            perms[:, b] .= sortperm(norms; rev = true, alg = MergeSort)
        end
        vec([perms[k, b] + (b - 1) * M_rel for k in 1:M_rel, b in 1:B])
    end
    q_s = reshape(reshape(q_rel, d, M_rel * B)[:, lin_idx], d, M_rel, B)
    p_s = reshape(reshape(p_rel, d, M_rel * B)[:, lin_idx], d, M_rel, B)
    (q = q_s, p = p_s)
end

@doc raw"""
    CanonicalSlotLayer{M, NSlots}()
    CanonicalSlotLayer(M::Integer, NSlots::Integer)

Make an instance of a canonical slot reduction layer mapping `M` particles to `NSlots` collective canonical slots (`M >= 2`, `1 <= NSlots <= M`).

The layer applies the orthogonal Helmert / Jacobi transformation separating the center-of-mass mode

```math
(Q_{\mathrm{cm}}, P_{\mathrm{cm}}) = \frac{1}{\sqrt{M}}\sum_{i=1}^M (q_i, p_i)
```
from the ``M - 1`` zero-sum relative modes ``(q_{\mathrm{rel}}, p_{\mathrm{rel}}) \in \mathbb{R}^{d \times (M - 1)}``, sorts the ``M - 1`` relative modes canonically by descending squared phase-space norm ``\|q_{\mathrm{rel}, k}\|_2^2 + \|p_{\mathrm{rel}, k}\|_2^2`` (`sortperm` with `rev = true, alg = MergeSort`, applying the identical permutation ``\sigma_{\mathrm{rel}} \in S_{M - 1}`` to both ``q_{\mathrm{rel}}`` and ``p_{\mathrm{rel}}``), and projects the sorted relative modes onto ``N_{\mathrm{slots}} - 1`` relative slots via a learned [`StiefelManifold`](@extref GeometricOptimizers GeometricOptimizers.StiefelManifold) parameter ``\Phi_{\mathrm{rel}} \in \mathrm{St}(N_{\mathrm{slots}} - 1, M - 1)`` while keeping ``(Q_{\mathrm{cm}}, P_{\mathrm{cm}})`` intact as slot 1.
"""
struct CanonicalSlotLayer{M, NSlots} <: AbstractExplicitLayer{M, NSlots}
    function CanonicalSlotLayer{M, NSlots}() where {M, NSlots}
        (M isa Integer && NSlots isa Integer && M >= 2 && 1 <= NSlots <= M) ||
            throw(ArgumentError(
                "CanonicalSlotLayer requires M >= 2 and 1 <= NSlots <= M, got M = $M, NSlots = $NSlots."))
        new{Int(M), Int(NSlots)}()
    end
end

function CanonicalSlotLayer(M::Integer, NSlots::Integer)
    CanonicalSlotLayer{Int(M), Int(NSlots)}()
end

function parameterlength(::CanonicalSlotLayer{M, NSlots}) where {M, NSlots}
    K = NSlots - 1
    K == 0 ? 0 : (M - 1) * K - (K * (K + 1)) ÷ 2
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::CanonicalSlotLayer{M, NSlots}, backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, NSlots, T}
    K = NSlots - 1
    K == 0 && return NamedTuple()
    raw = KernelAbstractions.allocate(backend, T, M - 1, K)
    Φ_rel = StiefelManifold(orthonormal_columns() do
        initializer(rng, raw)
        raw
    end)
    (Φ_rel = Φ_rel,)
end

function _encode_slots_2d(
        ::CanonicalSlotLayer{M, NSlots}, q::AbstractMatrix{T}, p::AbstractMatrix{T},
        ps::NamedTuple) where {M, NSlots, T}
    scale_cm = inv(sqrt(T(M)))
    q_cm = sum(q; dims = 2) .* scale_cm
    p_cm = sum(p; dims = 2) .* scale_cm
    if NSlots == 1
        return (q = q_cm, p = p_cm)
    end
    H_rel = ChainRulesCore.ignore_derivatives(() -> _helmert_relative_matrix(T, M))
    q_rel = q * H_rel
    p_rel = p * H_rel
    sorted_rel = _sort_relative_modes_2d(q_rel, p_rel)
    q_slot_rel = sorted_rel.q * ps.Φ_rel
    p_slot_rel = sorted_rel.p * ps.Φ_rel
    (q = hcat(q_cm, q_slot_rel), p = hcat(p_cm, p_slot_rel))
end

function _encode_slots_3d(
        ::CanonicalSlotLayer{M, NSlots}, q::AbstractArray{T, 3}, p::AbstractArray{T, 3},
        ps::NamedTuple) where {M, NSlots, T}
    d, _, B = size(q)
    scale_cm = inv(sqrt(T(M)))
    q_cm = sum(q; dims = 2) .* scale_cm
    p_cm = sum(p; dims = 2) .* scale_cm
    if NSlots == 1
        return (q = q_cm, p = p_cm)
    end
    M_rel = M - 1
    K = NSlots - 1
    H_rel = ChainRulesCore.ignore_derivatives(() -> _helmert_relative_matrix(T, M))
    q_2d = reshape(permutedims(q, (1, 3, 2)), d * B, M)
    p_2d = reshape(permutedims(p, (1, 3, 2)), d * B, M)
    q_rel = permutedims(reshape(q_2d * H_rel, d, B, M_rel), (1, 3, 2))
    p_rel = permutedims(reshape(p_2d * H_rel, d, B, M_rel), (1, 3, 2))
    sorted_rel = _sort_relative_modes_3d(q_rel, p_rel)
    q_rel_s_2d = reshape(permutedims(sorted_rel.q, (1, 3, 2)), d * B, M_rel)
    p_rel_s_2d = reshape(permutedims(sorted_rel.p, (1, 3, 2)), d * B, M_rel)
    q_slot_rel = permutedims(reshape(q_rel_s_2d * ps.Φ_rel, d, B, K), (1, 3, 2))
    p_slot_rel = permutedims(reshape(p_rel_s_2d * ps.Φ_rel, d, B, K), (1, 3, 2))
    (q = cat(q_cm, q_slot_rel; dims = 2), p = cat(p_cm, p_slot_rel; dims = 2))
end

function (l::CanonicalSlotLayer{M, NSlots})(
        qp::NamedTuple{(:q, :p)}, ps::NamedTuple) where {M, NSlots}
    q, p = qp.q, qp.p
    nd = ndims(q)
    (nd in (2, 3) && size(q) == size(p) && size(q, 2) == M && size(q, 1) >= 1) ||
        throw(DimensionMismatch(
            "CanonicalSlotLayer{$M, $NSlots} expects q and p of matching size (d, $M) or (d, $M, B) with d >= 1, got $(size(q)) and $(size(p))."))
    nd == 2 ? _encode_slots_2d(l, q, p, ps) : _encode_slots_3d(l, q, p, ps)
end

function (l::CanonicalSlotLayer{M, NSlots})(
        z::AbstractVector, ps::NamedTuple) where {M, NSlots}
    N2 = length(z)
    (N2 >= 2 * M && mod(N2, 2 * M) == 0) ||
        throw(DimensionMismatch(
            "CanonicalSlotLayer{$M, $NSlots} expects first dimension divisible by $(2 * M) with d >= 1, got $N2."))
    d = N2 ÷ (2 * M)
    N = M * d
    q = reshape(@view(z[1:N]), d, M)
    p = reshape(@view(z[(N + 1):N2]), d, M)
    out = l((q = q, p = p), ps)
    vcat(vec(out.q), vec(out.p))
end

function (l::CanonicalSlotLayer{M, NSlots})(
        z::AbstractMatrix, ps::NamedTuple) where {M, NSlots}
    N2, B = size(z)
    (N2 >= 2 * M && mod(N2, 2 * M) == 0) ||
        throw(DimensionMismatch(
            "CanonicalSlotLayer{$M, $NSlots} expects first dimension divisible by $(2 * M) with d >= 1, got $N2."))
    d = N2 ÷ (2 * M)
    N = M * d
    N_out = NSlots * d
    q = reshape(@view(z[1:N, :]), d, M, B)
    p = reshape(@view(z[(N + 1):N2, :]), d, M, B)
    out = l((q = q, p = p), ps)
    vcat(reshape(out.q, N_out, B), reshape(out.p, N_out, B))
end

@doc raw"""
    InverseCanonicalSlotLayer{M, NSlots}()
    InverseCanonicalSlotLayer(M::Integer, NSlots::Integer)

Analytic symplectic decoder lift (`inverse_layer(CanonicalSlotLayer(M, NSlots))`) mapping `NSlots` collective canonical slots back to `M` particles via ``\Phi_{\mathrm{rel}} \in \mathrm{St}(N_{\mathrm{slots}} - 1, M - 1)`` and the inverse Helmert matrix.

For any ``\Phi_{\mathrm{rel}} \in \mathrm{St}(N_{\mathrm{slots}} - 1, M - 1)``, the decoder lift satisfies ``(\nabla \Psi_{\mathrm{slot}}^{\mathrm{dec}})^T \mathbb{J}_{2Md} \nabla \Psi_{\mathrm{slot}}^{\mathrm{dec}} = \mathbb{J}_{2 N_{\mathrm{slots}} d}`` and preserves ``(Q_{\mathrm{cm}}, P_{\mathrm{cm}})``.
"""
struct InverseCanonicalSlotLayer{M, NSlots} <: AbstractExplicitLayer{NSlots, M}
    function InverseCanonicalSlotLayer{M, NSlots}() where {M, NSlots}
        (M isa Integer && NSlots isa Integer && M >= 2 && 1 <= NSlots <= M) ||
            throw(ArgumentError(
                "InverseCanonicalSlotLayer requires M >= 2 and 1 <= NSlots <= M, got M = $M, NSlots = $NSlots."))
        new{Int(M), Int(NSlots)}()
    end
end

function InverseCanonicalSlotLayer(M::Integer, NSlots::Integer)
    InverseCanonicalSlotLayer{Int(M), Int(NSlots)}()
end

function parameterlength(::InverseCanonicalSlotLayer{M, NSlots}) where {M, NSlots}
    parameterlength(CanonicalSlotLayer{M, NSlots}())
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::InverseCanonicalSlotLayer{M, NSlots}, backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, NSlots, T}
    initialparameters(rng, initializer, CanonicalSlotLayer{M, NSlots}(), backend, T)
end

function _decode_slots_2d(
        ::InverseCanonicalSlotLayer{M, NSlots}, q_slot::AbstractMatrix{T},
        p_slot::AbstractMatrix{T}, ps::NamedTuple) where {M, NSlots, T}
    scale_cm = inv(sqrt(T(M)))
    ones_row = ChainRulesCore.ignore_derivatives(() -> fill(scale_cm, 1, M))
    q_cm_lift = @view(q_slot[:, 1:1]) * ones_row
    p_cm_lift = @view(p_slot[:, 1:1]) * ones_row
    if NSlots == 1
        return (q = q_cm_lift, p = p_cm_lift)
    end
    H_rel = ChainRulesCore.ignore_derivatives(() -> _helmert_relative_matrix(T, M))
    W_rel_T = transpose(H_rel * ps.Φ_rel)
    q_rel_lift = @view(q_slot[:, 2:NSlots]) * W_rel_T
    p_rel_lift = @view(p_slot[:, 2:NSlots]) * W_rel_T
    (q = q_cm_lift .+ q_rel_lift, p = p_cm_lift .+ p_rel_lift)
end

function _decode_slots_3d(
        ::InverseCanonicalSlotLayer{M, NSlots}, q_slot::AbstractArray{T, 3},
        p_slot::AbstractArray{T, 3}, ps::NamedTuple) where {M, NSlots, T}
    d, _, B = size(q_slot)
    scale_cm = inv(sqrt(T(M)))
    ones_row = ChainRulesCore.ignore_derivatives(() -> fill(scale_cm, 1, M, 1))
    q_cm_lift = @view(q_slot[:, 1:1, :]) .* ones_row
    p_cm_lift = @view(p_slot[:, 1:1, :]) .* ones_row
    if NSlots == 1
        return (q = q_cm_lift, p = p_cm_lift)
    end
    K = NSlots - 1
    H_rel = ChainRulesCore.ignore_derivatives(() -> _helmert_relative_matrix(T, M))
    W_rel_T = transpose(H_rel * ps.Φ_rel)
    q_rel_2d = reshape(permutedims(@view(q_slot[:, 2:NSlots, :]), (1, 3, 2)), d * B, K)
    p_rel_2d = reshape(permutedims(@view(p_slot[:, 2:NSlots, :]), (1, 3, 2)), d * B, K)
    q_rel_lift = permutedims(reshape(q_rel_2d * W_rel_T, d, B, M), (1, 3, 2))
    p_rel_lift = permutedims(reshape(p_rel_2d * W_rel_T, d, B, M), (1, 3, 2))
    (q = q_cm_lift .+ q_rel_lift, p = p_cm_lift .+ p_rel_lift)
end

function (l::InverseCanonicalSlotLayer{M, NSlots})(
        qp::NamedTuple{(:q, :p)}, ps::NamedTuple) where {M, NSlots}
    q, p = qp.q, qp.p
    nd = ndims(q)
    (nd in (2, 3) && size(q) == size(p) && size(q, 2) == NSlots && size(q, 1) >= 1) ||
        throw(DimensionMismatch(
            "InverseCanonicalSlotLayer{$M, $NSlots} expects q and p of matching size (d, $NSlots) or (d, $NSlots, B) with d >= 1, got $(size(q)) and $(size(p))."))
    nd == 2 ? _decode_slots_2d(l, q, p, ps) : _decode_slots_3d(l, q, p, ps)
end

function (l::InverseCanonicalSlotLayer{M, NSlots})(
        z::AbstractVector, ps::NamedTuple) where {M, NSlots}
    N2 = length(z)
    (N2 >= 2 * NSlots && mod(N2, 2 * NSlots) == 0) ||
        throw(DimensionMismatch(
            "InverseCanonicalSlotLayer{$M, $NSlots} expects first dimension divisible by $(2 * NSlots) with d >= 1, got $N2."))
    d = N2 ÷ (2 * NSlots)
    N_in = NSlots * d
    q = reshape(@view(z[1:N_in]), d, NSlots)
    p = reshape(@view(z[(N_in + 1):N2]), d, NSlots)
    out = l((q = q, p = p), ps)
    vcat(vec(out.q), vec(out.p))
end

function (l::InverseCanonicalSlotLayer{M, NSlots})(
        z::AbstractMatrix, ps::NamedTuple) where {M, NSlots}
    N2, B = size(z)
    (N2 >= 2 * NSlots && mod(N2, 2 * NSlots) == 0) ||
        throw(DimensionMismatch(
            "InverseCanonicalSlotLayer{$M, $NSlots} expects first dimension divisible by $(2 * NSlots) with d >= 1, got $N2."))
    d = N2 ÷ (2 * NSlots)
    N_in = NSlots * d
    N_out = M * d
    q = reshape(@view(z[1:N_in, :]), d, NSlots, B)
    p = reshape(@view(z[(N_in + 1):N2, :]), d, NSlots, B)
    out = l((q = q, p = p), ps)
    vcat(reshape(out.q, N_out, B), reshape(out.p, N_out, B))
end

function inverse_layer(::CanonicalSlotLayer{M, NSlots}) where {M, NSlots}
    InverseCanonicalSlotLayer{M, NSlots}()
end
function inverse_layer(::InverseCanonicalSlotLayer{M, NSlots}) where {M, NSlots}
    CanonicalSlotLayer{M, NSlots}()
end
function inverse_layer(::Type{CanonicalSlotLayer{M, NSlots}}) where {M, NSlots}
    InverseCanonicalSlotLayer{M, NSlots}
end
function inverse_layer(::Type{InverseCanonicalSlotLayer{M, NSlots}}) where {M, NSlots}
    CanonicalSlotLayer{M, NSlots}
end
inverse_layer(::Type{CanonicalSlotLayer}) = InverseCanonicalSlotLayer
inverse_layer(::Type{InverseCanonicalSlotLayer}) = CanonicalSlotLayer
