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

function (::FastDLRATraceLoss)(
        x::NamedTuple{(:X, :S, :V)}, x_hat::NamedTuple{(:X, :S, :V)})
    if ndims(x.X) == 2 && ndims(x_hat.X) == 2
        return _dlra_trace_loss_2d(x.X, x.S, x.V, x_hat.X, x_hat.S, x_hat.V)
    elseif ndims(x.X) == 3 && ndims(x_hat.X) == 3
        Nx = size(x.X, 1)
        size(x_hat.X, 1) == Nx ||
            throw(DimensionMismatch("X and X_hat must have the same number of rows; got $Nx and $(size(x_hat.X, 1))"))
        Nv = size(x.V, 1)
        size(x_hat.V, 1) == Nv ||
            throw(DimensionMismatch("V and V_hat must have the same number of rows; got $Nv and $(size(x_hat.V, 1))"))
        _, B = _validate_dlra_factors(x.X, x.S, x.V, Nx, Nv)
        _, B_hat = _validate_dlra_factors(x_hat.X, x_hat.S, x_hat.V, Nx, Nv)
        B == B_hat ||
            throw(DimensionMismatch("batch sizes of x and x_hat must match; got $B and $B_hat"))
        T = promote_type(
            eltype(x.X), eltype(x.S), eltype(x.V),
            eltype(x_hat.X), eltype(x_hat.S), eltype(x_hat.V))
        total = zero(T)
        for b in 1:B
            total += _dlra_trace_loss_2d(
                @view(x.X[:, :, b]), @view(x.S[:, :, b]), @view(x.V[:, :, b]),
                @view(x_hat.X[:, :, b]), @view(x_hat.S[:, :, b]), @view(x_hat.V[:, :, b]))
        end
        return total
    else
        throw(DimensionMismatch("DLRA factors x and x_hat must both be 2D or both 3D arrays"))
    end
end

function (loss::FastDLRATraceLoss)(
        model::Union{Chain, AbstractExplicitLayer},
        ps,
        input::NamedTuple{(:X, :S, :V)},
        output::NamedTuple{(:X, :S, :V)})
    return loss(model(input, ps), output)
end
