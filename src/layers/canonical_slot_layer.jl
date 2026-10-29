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
    CanonicalSlotLayer{N, NOut, M, NSlots}()
    CanonicalSlotLayer(M::Integer, d::Integer, NSlots::Integer)

Make an instance of a canonical slot reduction layer mapping `M` particles in ``d`` spatial dimensions to `NSlots` collective canonical slots (`M >= 2`, `1 <= NSlots <= M`).

The first two type parameters are the phase-space dimensions `N = 2Md` of the particles and `NOut = 2 NSlots d` of the slots, which `input_dimension` and `output_dimension` return.
The layer first sorts the particles along coordinate row 1 of `q` with [`CanonicalSortLayer`](@ref), so that its output is invariant under a relabelling of particles with distinct first coordinates. It then applies the orthogonal Helmert / Jacobi transformation separating the center-of-mass mode

```math
(Q_{\mathrm{cm}}, P_{\mathrm{cm}}) = \frac{1}{\sqrt{M}}\sum_{i=1}^M (q_i, p_i)
```
from the ``M - 1`` zero-sum relative modes ``(q_{\mathrm{rel}}, p_{\mathrm{rel}}) \in \mathbb{R}^{d \times (M - 1)}``, sorts the ``M - 1`` relative modes canonically by descending squared phase-space norm ``\|q_{\mathrm{rel}, k}\|_2^2 + \|p_{\mathrm{rel}, k}\|_2^2`` (`sortperm` with `rev = true, alg = MergeSort`, applying the identical permutation ``\sigma_{\mathrm{rel}} \in S_{M - 1}`` to both ``q_{\mathrm{rel}}`` and ``p_{\mathrm{rel}}``), and projects the sorted relative modes onto ``N_{\mathrm{slots}} - 1`` relative slots via a learned [`StiefelManifold`](@extref GeometricOptimizers GeometricOptimizers.StiefelManifold) parameter ``\Phi_{\mathrm{rel}} \in \mathrm{St}(N_{\mathrm{slots}} - 1, M - 1)`` while keeping ``(Q_{\mathrm{cm}}, P_{\mathrm{cm}})`` intact as slot 1.
"""
struct CanonicalSlotLayer{N, NOut, M, NSlots} <: AbstractExplicitLayer{N, NOut}
    function CanonicalSlotLayer{N, NOut, M, NSlots}() where {N, NOut, M, NSlots}
        _check_slot_params("CanonicalSlotLayer", N, NOut, M, NSlots)
        new{Int(N), Int(NOut), Int(M), Int(NSlots)}()
    end
end

function _check_slot_params(name::String, N, NOut, M, NSlots)
    (N isa Integer && NOut isa Integer && M isa Integer && NSlots isa Integer &&
     M >= 2 && 1 <= NSlots <= M && N >= 2 * M && mod(N, 2 * M) == 0 &&
     NOut == 2 * NSlots * (N ÷ (2 * M))) ||
        throw(ArgumentError(
            "$name{N, NOut, M, NSlots} requires M >= 2, 1 <= NSlots <= M, N = 2Md and NOut = 2 NSlots d with d >= 1, got N = $N, NOut = $NOut, M = $M, NSlots = $NSlots."))
    nothing
end

function _slot_type_parameters(name::String, M::Integer, d::Integer, NSlots::Integer)
    (M >= 2 && d >= 1 && 1 <= NSlots <= M) ||
        throw(ArgumentError(
            "$name requires M >= 2, d >= 1 and 1 <= NSlots <= M, got M = $M, d = $d, NSlots = $NSlots."))
    (2 * Int(M) * Int(d), 2 * Int(NSlots) * Int(d), Int(M), Int(NSlots))
end

function CanonicalSlotLayer(M::Integer, d::Integer, NSlots::Integer)
    CanonicalSlotLayer{_slot_type_parameters("CanonicalSlotLayer", M, d, NSlots)...}()
end

function parameterlength(::CanonicalSlotLayer{
        N, NOut, M, NSlots}) where {
        N, NOut, M, NSlots}
    K = NSlots - 1
    K == 0 ? 0 : (M - 1) * K - (K * (K + 1)) ÷ 2
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::CanonicalSlotLayer{N, NOut, M, NSlots}, backend::KernelAbstractions.Backend,
        ::Type{T}) where {N, NOut, M, NSlots, T}
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
        ::CanonicalSlotLayer{N, NOut, M, NSlots}, q::AbstractMatrix{T},
        p::AbstractMatrix{T}, ps::NamedTuple) where {N, NOut, M, NSlots, T}
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
        ::CanonicalSlotLayer{N, NOut, M, NSlots}, q::AbstractArray{T, 3},
        p::AbstractArray{T, 3}, ps::NamedTuple) where {N, NOut, M, NSlots, T}
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

function (l::CanonicalSlotLayer{N, NOut, M, NSlots})(
        qp::NamedTuple{(:q, :p)}, ps::NamedTuple) where {N, NOut, M, NSlots}
    q, p = qp.q, qp.p
    d = N ÷ (2 * M)
    (ndims(q) in (2, 3) && size(q) == size(p) && size(q, 1) == d && size(q, 2) == M) ||
        throw(DimensionMismatch(
            "CanonicalSlotLayer{$N, $NOut, $M, $NSlots} expects q and p of matching size ($d, $M) or ($d, $M, B), got $(size(q)) and $(size(p))."))
    sorted = CanonicalSortLayer{N, M, 1}()(qp)
    ndims(q) == 2 ? _encode_slots_2d(l, sorted.q, sorted.p, ps) :
    _encode_slots_3d(l, sorted.q, sorted.p, ps)
end

function (l::CanonicalSlotLayer{N, NOut, M, NSlots})(
        z::AbstractVector, ps::NamedTuple) where {N, NOut, M, NSlots}
    _check_first_dim("CanonicalSlotLayer{$N, $NOut, $M, $NSlots}", N, length(z))
    d = N ÷ (2 * M)
    Nq = M * d
    q = reshape(@view(z[1:Nq]), d, M)
    p = reshape(@view(z[(Nq + 1):N]), d, M)
    out = l((q = q, p = p), ps)
    vcat(vec(out.q), vec(out.p))
end

function (l::CanonicalSlotLayer{N, NOut, M, NSlots})(
        z::AbstractMatrix, ps::NamedTuple) where {N, NOut, M, NSlots}
    _check_first_dim("CanonicalSlotLayer{$N, $NOut, $M, $NSlots}", N, size(z, 1))
    B = size(z, 2)
    d = N ÷ (2 * M)
    Nq = M * d
    q = reshape(@view(z[1:Nq, :]), d, M, B)
    p = reshape(@view(z[(Nq + 1):N, :]), d, M, B)
    out = l((q = q, p = p), ps)
    vcat(reshape(out.q, NOut ÷ 2, B), reshape(out.p, NOut ÷ 2, B))
end

@doc raw"""
    InverseCanonicalSlotLayer{N, NOut, M, NSlots}()
    InverseCanonicalSlotLayer(M::Integer, d::Integer, NSlots::Integer)

Analytic symplectic decoder lift (`inverse_layer(CanonicalSlotLayer(M, d, NSlots))`) mapping `NSlots` collective canonical slots back to `M` particles via ``\Phi_{\mathrm{rel}} \in \mathrm{St}(N_{\mathrm{slots}} - 1, M - 1)`` and the inverse Helmert matrix.

It takes the type parameters of [`CanonicalSlotLayer`](@ref), so its input dimension is `NOut = 2 NSlots d` and its output dimension is `N = 2Md`.
For any ``\Phi_{\mathrm{rel}} \in \mathrm{St}(N_{\mathrm{slots}} - 1, M - 1)``, the decoder lift satisfies ``(\nabla \Psi_{\mathrm{slot}}^{\mathrm{dec}})^T \mathbb{J}_{2Md} \nabla \Psi_{\mathrm{slot}}^{\mathrm{dec}} = \mathbb{J}_{2 N_{\mathrm{slots}} d}`` and preserves ``(Q_{\mathrm{cm}}, P_{\mathrm{cm}})``.
"""
struct InverseCanonicalSlotLayer{N, NOut, M, NSlots} <: AbstractExplicitLayer{NOut, N}
    function InverseCanonicalSlotLayer{N, NOut, M, NSlots}() where {N, NOut, M, NSlots}
        _check_slot_params("InverseCanonicalSlotLayer", N, NOut, M, NSlots)
        new{Int(N), Int(NOut), Int(M), Int(NSlots)}()
    end
end

function InverseCanonicalSlotLayer(M::Integer, d::Integer, NSlots::Integer)
    InverseCanonicalSlotLayer{
        _slot_type_parameters("InverseCanonicalSlotLayer", M, d, NSlots)...}()
end

function parameterlength(::InverseCanonicalSlotLayer{
        N, NOut, M, NSlots}) where {
        N, NOut, M, NSlots}
    parameterlength(CanonicalSlotLayer{N, NOut, M, NSlots}())
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::InverseCanonicalSlotLayer{N, NOut, M, NSlots},
        backend::KernelAbstractions.Backend,
        ::Type{T}) where {N, NOut, M, NSlots, T}
    initialparameters(
        rng, initializer, CanonicalSlotLayer{N, NOut, M, NSlots}(), backend, T)
end

function _decode_slots_2d(
        ::InverseCanonicalSlotLayer{N, NOut, M, NSlots}, q_slot::AbstractMatrix{T},
        p_slot::AbstractMatrix{T}, ps::NamedTuple) where {N, NOut, M, NSlots, T}
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
        ::InverseCanonicalSlotLayer{N, NOut, M, NSlots}, q_slot::AbstractArray{T, 3},
        p_slot::AbstractArray{T, 3}, ps::NamedTuple) where {N, NOut, M, NSlots, T}
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

function (l::InverseCanonicalSlotLayer{N, NOut, M, NSlots})(
        qp::NamedTuple{(:q, :p)}, ps::NamedTuple) where {N, NOut, M, NSlots}
    q, p = qp.q, qp.p
    d = N ÷ (2 * M)
    (ndims(q) in (2, 3) && size(q) == size(p) && size(q, 1) == d &&
     size(q, 2) == NSlots) ||
        throw(DimensionMismatch(
            "InverseCanonicalSlotLayer{$N, $NOut, $M, $NSlots} expects q and p of matching size ($d, $NSlots) or ($d, $NSlots, B), got $(size(q)) and $(size(p))."))
    ndims(q) == 2 ? _decode_slots_2d(l, q, p, ps) : _decode_slots_3d(l, q, p, ps)
end

function (l::InverseCanonicalSlotLayer{N, NOut, M, NSlots})(
        z::AbstractVector, ps::NamedTuple) where {N, NOut, M, NSlots}
    _check_first_dim("InverseCanonicalSlotLayer{$N, $NOut, $M, $NSlots}", NOut, length(z))
    d = N ÷ (2 * M)
    N_in = NSlots * d
    q = reshape(@view(z[1:N_in]), d, NSlots)
    p = reshape(@view(z[(N_in + 1):NOut]), d, NSlots)
    out = l((q = q, p = p), ps)
    vcat(vec(out.q), vec(out.p))
end

function (l::InverseCanonicalSlotLayer{N, NOut, M, NSlots})(
        z::AbstractMatrix, ps::NamedTuple) where {N, NOut, M, NSlots}
    _check_first_dim("InverseCanonicalSlotLayer{$N, $NOut, $M, $NSlots}", NOut, size(z, 1))
    B = size(z, 2)
    d = N ÷ (2 * M)
    N_in = NSlots * d
    q = reshape(@view(z[1:N_in, :]), d, NSlots, B)
    p = reshape(@view(z[(N_in + 1):NOut, :]), d, NSlots, B)
    out = l((q = q, p = p), ps)
    vcat(reshape(out.q, N ÷ 2, B), reshape(out.p, N ÷ 2, B))
end

function inverse_layer(::CanonicalSlotLayer{N, NOut, M, NSlots}) where {N, NOut, M, NSlots}
    InverseCanonicalSlotLayer{N, NOut, M, NSlots}()
end
function inverse_layer(::InverseCanonicalSlotLayer{
        N, NOut, M, NSlots}) where {
        N, NOut, M, NSlots}
    CanonicalSlotLayer{N, NOut, M, NSlots}()
end
function inverse_layer(::Type{CanonicalSlotLayer{
        N, NOut, M, NSlots}}) where {
        N, NOut, M, NSlots}
    InverseCanonicalSlotLayer{N, NOut, M, NSlots}
end
function inverse_layer(::Type{InverseCanonicalSlotLayer{
        N, NOut, M, NSlots}}) where {
        N, NOut, M, NSlots}
    CanonicalSlotLayer{N, NOut, M, NSlots}
end
inverse_layer(::Type{CanonicalSlotLayer}) = InverseCanonicalSlotLayer
inverse_layer(::Type{InverseCanonicalSlotLayer}) = CanonicalSlotLayer
