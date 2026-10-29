@doc raw"""
    FastDLRATraceLoss()

Construct a gauge-invariant low-rank trace loss that evaluates the squared Frobenius
reconstruction error ``\|X S V^T - \hat{X} \hat{S} \hat{V}^T\|_F^2`` between two factored
dynamical low-rank approximation (DLRA) states `(X = X, S = S, V = V)` and
`(X = X_hat, S = S_hat, V = V_hat)` in ``O(r^2 (N_x + N_v))`` operations without forming
any dense ``N_x \times N_v`` array:

```math
\|X S V^T - \hat{X} \hat{S} \hat{V}^T\|_F^2 =
\mathrm{tr}\bigl(S^T (X^T X) S (V^T V)\bigr)
- 2\,\mathrm{tr}\bigl(S^T (X^T \hat{X}) \hat{S} (\hat{V}^T V)\bigr)
+ \mathrm{tr}\bigl(\hat{S}^T (\hat{X}^T \hat{X}) \hat{S} (\hat{V}^T \hat{V})\bigr),
```

clamped at `zero(T)` via `max(zero(T), val)` to prevent negative roundoff at zero error.
Because the Gram matrices ``X^T X, V^T V, \hat{X}^T \hat{X}, \hat{V}^T \hat{V}`` are
computed explicitly, the identity holds even when the basis factors are only
approximately orthonormal.

See also [`LowRankStiefelLayer`](@ref) and [`DecodeLowRankStiefelLayer`](@ref).
"""
struct FastDLRATraceLoss <: NetworkLoss end

function _dlra_trace_loss_2d(
        X::AbstractMatrix, S::AbstractMatrix, V::AbstractMatrix,
        X_hat::AbstractMatrix, S_hat::AbstractMatrix, V_hat::AbstractMatrix)
    Nx = size(X, 1)
    size(X_hat, 1) == Nx ||
        throw(DimensionMismatch("X and X_hat must have the same number of rows; got $Nx and $(size(X_hat, 1))"))
    Nv = size(V, 1)
    size(V_hat, 1) == Nv ||
        throw(DimensionMismatch("V and V_hat must have the same number of rows; got $Nv and $(size(V_hat, 1))"))
    _validate_dlra_factors(X, S, V, Nx, Nv)
    _validate_dlra_factors(X_hat, S_hat, V_hat, Nx, Nv)

    T = promote_type(
        eltype(X), eltype(S), eltype(V), eltype(X_hat), eltype(S_hat), eltype(V_hat))
    Gx11 = X' * X
    Gv11 = V' * V
    Gx22 = X_hat' * X_hat
    Gv22 = V_hat' * V_hat
    Gx12 = X' * X_hat
    Gv21 = V_hat' * V

    term1 = tr((S' * Gx11 * S) * Gv11)
    term2 = tr((S_hat' * Gx22 * S_hat) * Gv22)
    cross = tr((S' * Gx12 * S_hat) * Gv21)
    val = T(term1 - 2 * cross + term2)
    return max(zero(T), val)
end

function _batched_at_b(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where {T}
    m, k, batch = size(A, 2), size(B, 2), size(A, 3)
    C = similar(A, T, m, k, batch)
    @inbounds for b in 1:batch
        mul!(@view(C[:, :, b]), transpose(@view(A[:, :, b])), @view(B[:, :, b]))
    end
    return C
end

function _batched_a_b(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where {T}
    m, k, batch = size(A, 1), size(B, 2), size(A, 3)
    C = similar(A, T, m, k, batch)
    @inbounds for b in 1:batch
        mul!(@view(C[:, :, b]), @view(A[:, :, b]), @view(B[:, :, b]))
    end
    return C
end

function _batched_a_bt(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where {T}
    m, k, batch = size(A, 1), size(B, 1), size(A, 3)
    C = similar(A, T, m, k, batch)
    @inbounds for b in 1:batch
        mul!(@view(C[:, :, b]), @view(A[:, :, b]), transpose(@view(B[:, :, b])))
    end
    return C
end

function _dlra_trace_loss_3d(
        X::AbstractArray{<:Any, 3}, S::AbstractArray{<:Any, 3}, V::AbstractArray{<:Any, 3},
        X_hat::AbstractArray{<:Any, 3}, S_hat::AbstractArray{<:Any, 3}, V_hat::AbstractArray{
            <:Any, 3})
    Nx = size(X, 1)
    size(X_hat, 1) == Nx ||
        throw(DimensionMismatch("X and X_hat must have the same number of rows; got $Nx and $(size(X_hat, 1))"))
    Nv = size(V, 1)
    size(V_hat, 1) == Nv ||
        throw(DimensionMismatch("V and V_hat must have the same number of rows; got $Nv and $(size(V_hat, 1))"))
    _, B = _validate_dlra_factors(X, S, V, Nx, Nv)
    _, B_hat = _validate_dlra_factors(X_hat, S_hat, V_hat, Nx, Nv)
    B == B_hat ||
        throw(DimensionMismatch("batch sizes of x and x_hat must match; got $B and $B_hat"))

    T = promote_type(
        eltype(X), eltype(S), eltype(V),
        eltype(X_hat), eltype(S_hat), eltype(V_hat))
    X_T = eltype(X) === T ? X : T.(X)
    S_T = eltype(S) === T ? S : T.(S)
    V_T = eltype(V) === T ? V : T.(V)
    Xh_T = eltype(X_hat) === T ? X_hat : T.(X_hat)
    Sh_T = eltype(S_hat) === T ? S_hat : T.(S_hat)
    Vh_T = eltype(V_hat) === T ? V_hat : T.(V_hat)

    Gx11 = _batched_at_b(X_T, X_T)
    Gv11 = _batched_at_b(V_T, V_T)
    Gx22 = _batched_at_b(Xh_T, Xh_T)
    Gv22 = _batched_at_b(Vh_T, Vh_T)
    Gx12 = _batched_at_b(X_T, Xh_T)
    Gv21 = _batched_at_b(Vh_T, V_T)

    M1 = _batched_a_b(_batched_at_b(S_T, Gx11), S_T)
    M2 = _batched_a_b(_batched_at_b(Sh_T, Gx22), Sh_T)
    Mc = _batched_a_b(_batched_at_b(S_T, Gx12), Sh_T)

    total = zero(T)
    @inbounds for b in 1:B
        t1 = zero(T)
        for j in axes(M1, 2), i in axes(M1, 1)

            t1 += M1[i, j, b] * Gv11[j, i, b]
        end
        t2 = zero(T)
        for j in axes(M2, 2), i in axes(M2, 1)

            t2 += M2[i, j, b] * Gv22[j, i, b]
        end
        tc = zero(T)
        for j in axes(Mc, 2), i in axes(Mc, 1)

            tc += Mc[i, j, b] * Gv21[j, i, b]
        end
        total += max(zero(T), t1 - 2 * tc + t2)
    end
    return total
end

function ChainRulesCore.rrule(
        ::typeof(_dlra_trace_loss_3d),
        X::AbstractArray{<:Any, 3}, S::AbstractArray{<:Any, 3}, V::AbstractArray{<:Any, 3},
        X_hat::AbstractArray{<:Any, 3}, S_hat::AbstractArray{<:Any, 3}, V_hat::AbstractArray{
            <:Any, 3})
    Nx = size(X, 1)
    size(X_hat, 1) == Nx ||
        throw(DimensionMismatch("X and X_hat must have the same number of rows; got $Nx and $(size(X_hat, 1))"))
    Nv = size(V, 1)
    size(V_hat, 1) == Nv ||
        throw(DimensionMismatch("V and V_hat must have the same number of rows; got $Nv and $(size(V_hat, 1))"))
    _, B = _validate_dlra_factors(X, S, V, Nx, Nv)
    _, B_hat = _validate_dlra_factors(X_hat, S_hat, V_hat, Nx, Nv)
    B == B_hat ||
        throw(DimensionMismatch("batch sizes of x and x_hat must match; got $B and $B_hat"))

    T = promote_type(
        eltype(X), eltype(S), eltype(V),
        eltype(X_hat), eltype(S_hat), eltype(V_hat))
    X_T = eltype(X) === T ? X : T.(X)
    S_T = eltype(S) === T ? S : T.(S)
    V_T = eltype(V) === T ? V : T.(V)
    Xh_T = eltype(X_hat) === T ? X_hat : T.(X_hat)
    Sh_T = eltype(S_hat) === T ? S_hat : T.(S_hat)
    Vh_T = eltype(V_hat) === T ? V_hat : T.(V_hat)

    Gx11 = _batched_at_b(X_T, X_T)
    Gv11 = _batched_at_b(V_T, V_T)
    Gx22 = _batched_at_b(Xh_T, Xh_T)
    Gv22 = _batched_at_b(Vh_T, Vh_T)
    Gx12 = _batched_at_b(X_T, Xh_T)
    Gv21 = _batched_at_b(Vh_T, V_T)

    M1 = _batched_a_b(_batched_at_b(S_T, Gx11), S_T)
    M2 = _batched_a_b(_batched_at_b(Sh_T, Gx22), Sh_T)
    Mc = _batched_a_b(_batched_at_b(S_T, Gx12), Sh_T)

    raw = similar(X_T, T, 1, 1, B)
    total = zero(T)
    @inbounds for b in 1:B
        t1 = zero(T)
        for j in axes(M1, 2), i in axes(M1, 1)

            t1 += M1[i, j, b] * Gv11[j, i, b]
        end
        t2 = zero(T)
        for j in axes(M2, 2), i in axes(M2, 1)

            t2 += M2[i, j, b] * Gv22[j, i, b]
        end
        tc = zero(T)
        for j in axes(Mc, 2), i in axes(Mc, 1)

            tc += Mc[i, j, b] * Gv21[j, i, b]
        end
        rb = t1 - 2 * tc + t2
        raw[1, 1, b] = rb
        total += max(zero(T), rb)
    end

    function _dlra_trace_loss_3d_pullback(Δ)
        Δy = ChainRulesCore.unthunk(Δ)
        if Δy isa ChainRulesCore.AbstractZero
            z = ChainRulesCore.ZeroTangent()
            return (ChainRulesCore.NoTangent(), z, z, z, z, z, z)
        end
        scale = (T(2 * Δy)) .* T.(raw .> zero(T))
        Gv12 = permutedims(Gv21, (2, 1, 3))
        Gx21 = permutedims(Gx12, (2, 1, 3))
        SGv11 = _batched_a_b(S_T, Gv11)
        ShGv21 = _batched_a_b(Sh_T, Gv21)
        ShGv22 = _batched_a_b(Sh_T, Gv22)
        SGv12 = _batched_a_b(S_T, Gv12)

        dX = scale .* (
            _batched_a_b(X_T, _batched_a_bt(SGv11, S_T)) .-
            _batched_a_b(Xh_T, _batched_a_bt(ShGv21, S_T)))
        dS = scale .* (
            _batched_a_b(Gx11, SGv11) .-
            _batched_a_b(Gx12, ShGv21))
        dV = scale .* (
            _batched_a_b(V_T, M1) .-
            _batched_a_bt(Vh_T, Mc))
        dX_hat = scale .* (
            _batched_a_b(Xh_T, _batched_a_bt(ShGv22, Sh_T)) .-
            _batched_a_b(X_T, _batched_a_bt(SGv12, Sh_T)))
        dS_hat = scale .* (
            _batched_a_b(Gx22, ShGv22) .-
            _batched_a_b(Gx21, SGv12))
        dV_hat = scale .* (
            _batched_a_b(Vh_T, M2) .-
            _batched_a_b(V_T, Mc))

        return (
            ChainRulesCore.NoTangent(),
            eltype(X) === T ? dX : eltype(X).(dX),
            eltype(S) === T ? dS : eltype(S).(dS),
            eltype(V) === T ? dV : eltype(V).(dV),
            eltype(X_hat) === T ? dX_hat : eltype(X_hat).(dX_hat),
            eltype(S_hat) === T ? dS_hat : eltype(S_hat).(dS_hat),
            eltype(V_hat) === T ? dV_hat : eltype(V_hat).(dV_hat)
        )
    end
    return total, _dlra_trace_loss_3d_pullback
end

function (::FastDLRATraceLoss)(
        x::NamedTuple{(:X, :S, :V)}, x_hat::NamedTuple{(:X, :S, :V)})
    ndims(x.X) == ndims(x.S) == ndims(x.V) ||
        throw(DimensionMismatch("DLRA factors X, S, and V of x must have the same number of dimensions; got ($(ndims(x.X)), $(ndims(x.S)), $(ndims(x.V)))"))
    ndims(x_hat.X) == ndims(x_hat.S) == ndims(x_hat.V) ||
        throw(DimensionMismatch("DLRA factors X, S, and V of x_hat must have the same number of dimensions; got ($(ndims(x_hat.X)), $(ndims(x_hat.S)), $(ndims(x_hat.V)))"))
    if ndims(x.X) == 2 && ndims(x_hat.X) == 2
        return _dlra_trace_loss_2d(x.X, x.S, x.V, x_hat.X, x_hat.S, x_hat.V)
    elseif ndims(x.X) == 3 && ndims(x_hat.X) == 3
        return _dlra_trace_loss_3d(x.X, x.S, x.V, x_hat.X, x_hat.S, x_hat.V)
    else
        throw(DimensionMismatch("DLRA factors x and x_hat must both be 2D or both 3D arrays"))
    end
end

function (loss::FastDLRATraceLoss)(
        model::Union{Chain, AbstractExplicitLayer},
        ps,
        input::NamedTuple{(:X, :S, :V)},
        output::NamedTuple{(:X, :S, :V)} = input)
    return loss(model(input, ps), output)
end

function (loss::FastDLRATraceLoss)(
        nn::NeuralNetwork,
        input::NamedTuple{(:X, :S, :V)},
        output::NamedTuple{(:X, :S, :V)} = input)
    return loss(nn.model, params(nn), input, output)
end
