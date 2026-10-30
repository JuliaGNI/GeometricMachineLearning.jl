@doc raw"""
    BilinearStiefelLayer(Nx, Nv, mx, mv; metric_factors = nothing)

Construct a bilinear Stiefel encoder layer that maps a phase-space grid field
``F \in \mathbb{R}^{N_x \times N_v}`` (or a batch ``F \in \mathbb{R}^{N_x \times N_v \times B}``)
to canonical coordinates ``(q, p)`` of half-dimension ``n = m_x m_v`` via a velocity-parity
split and two-sided Stiefel projection.

Given a phase-space array ``F``, the layer splits the velocity axis into time-reversal
even and odd components on the positive velocity half-grid ``1:(N_v \div 2)``,

```math
F^+ = \frac{1}{\sqrt{2}}\bigl(F_{:, 1:N_v/2} + F_{:, N_v:-1:N_v/2+1}\bigr), \qquad
F^- = \frac{1}{\sqrt{2}}\bigl(F_{:, 1:N_v/2} - F_{:, N_v:-1:N_v/2+1}\bigr),
```

and projects onto orthonormal bases ``\Phi_x \in \mathrm{St}(m_x, N_x)`` and
``\Phi_v \in \mathrm{St}(m_v, N_v / 2)``:

```math
Q_{\mathrm{red}} = \Phi_x^T F^+ \Phi_v \in \mathbb{R}^{m_x \times m_v}, \qquad
P_{\mathrm{red}} = \Phi_x^T F^- \Phi_v \in \mathbb{R}^{m_x \times m_v},
```

returning `(q = vec(Q_red), p = vec(P_red))` (or matrices of shape `(mx * mv, B)` for
3D batched input). A pre-split `NamedTuple` `(q = F_plus, p = F_minus)` of shape
`(Nx, Nv ÷ 2)` (or `(Nx, Nv ÷ 2, B)`) is also accepted directly.

When `metric_factors = (Rx, Rv)` is supplied with Cholesky or square-root mass factors
`Rx` of size `(Nx, Nx)` and `Rv` of size `(Nv ÷ 2, Nv ÷ 2)` or `(Nv, Nv)`, the field is
mapped into the Euclidean metric by `(Rx, Rv)` prior to Stiefel projection.

The coordinates ``(q, p)`` are canonical only for the structure ``\mathbb{J}_{2n}`` that the
layer and [`LowRankStiefelLayer`](@ref) declare on the parity coordinates. A frontend that is
linear in ``F`` is not a Poisson map for the Vlasov bracket: with ``L^2``-orthonormal even and
odd bases, the bracket linearised at ``f_0`` gives a singular ``\{Q, P\}``, not the identity.

See also [`TransposeBilinearStiefelLayer`](@ref), [`LowRankStiefelLayer`](@ref), and
[`inverse_layer`](@ref).
"""
struct BilinearStiefelLayer{Nx, Nv, Mx, Mv, MF} <: AbstractExplicitLayer{Nx, Mx}
    metric_factors::MF
end

@doc raw"""
    TransposeBilinearStiefelLayer(Nx, Nv, mx, mv; metric_factors = nothing)

Construct the right-inverse decoder layer corresponding to [`BilinearStiefelLayer`](@ref).

Given canonical coordinates `(q = q, p = p)` of length ``n = m_x m_v`` (or shape
`(mx * mv, B)` for batched input, or as a 2-tuple `(Q_red, P_red)` of shape `(mx, mv)`),
the layer reshapes ``(q, p)`` to
``Q_{\mathrm{red}}, P_{\mathrm{red}} \in \mathbb{R}^{m_x \times m_v}``, reconstructs the
even and odd velocity components

```math
F^+ = \Phi_x Q_{\mathrm{red}} \Phi_v^T, \qquad
F^- = \Phi_x P_{\mathrm{red}} \Phi_v^T,
```

applies ``(R_x^{-1}, R_v^{-1})`` when `metric_factors = (Rx, Rv)` is present, and
recombines the full velocity grid

```math
F_{:, 1:N_v/2} = \frac{1}{\sqrt{2}}(F^+ + F^-), \qquad
F_{:, N_v:-1:N_v/2+1} = \frac{1}{\sqrt{2}}(F^+ - F^-).
```

See also [`BilinearStiefelLayer`](@ref) and [`inverse_layer`](@ref).
"""
struct TransposeBilinearStiefelLayer{Nx, Nv, Mx, Mv, MF} <: AbstractExplicitLayer{Mx, Nx}
    metric_factors::MF
end

function _validate_bilinear_stiefel_dims(
        Nx::Integer, Nv::Integer, mx::Integer, mv::Integer, metric_factors)
    Nx >= 1 || throw(ArgumentError("Nx must be ≥ 1; got Nx = $Nx"))
    (Nv >= 2 && iseven(Nv)) ||
        throw(ArgumentError("Nv must be an even integer ≥ 2; got Nv = $Nv"))
    1 <= mx <= Nx ||
        throw(ArgumentError("mx must satisfy 1 ≤ mx ≤ Nx; got mx = $mx, Nx = $Nx"))
    half_Nv = Nv ÷ 2
    1 <= mv <= half_Nv ||
        throw(ArgumentError("mv must satisfy 1 ≤ mv ≤ Nv ÷ 2; got mv = $mv, Nv = $Nv"))
    if metric_factors !== nothing
        metric_factors isa Tuple{AbstractMatrix, AbstractMatrix} ||
            throw(ArgumentError("metric_factors must be a 2-tuple of matrices (Rx, Rv) or nothing"))
        Rx, Rv = metric_factors
        size(Rx) == (Nx, Nx) ||
            throw(DimensionMismatch("Rx must have size ($Nx, $Nx); got $(size(Rx))"))
        (size(Rv) == (half_Nv, half_Nv) || size(Rv) == (Nv, Nv)) ||
            throw(DimensionMismatch("Rv must have size ($half_Nv, $half_Nv) or ($Nv, $Nv); got $(size(Rv))"))
    end
    return nothing
end

function BilinearStiefelLayer(
        Nx::Integer, Nv::Integer, mx::Integer, mv::Integer; metric_factors = nothing)
    _validate_bilinear_stiefel_dims(Nx, Nv, mx, mv, metric_factors)
    return BilinearStiefelLayer{
        Int(Nx), Int(Nv), Int(mx), Int(mv), typeof(metric_factors)}(metric_factors)
end

function TransposeBilinearStiefelLayer(
        Nx::Integer, Nv::Integer, mx::Integer, mv::Integer; metric_factors = nothing)
    _validate_bilinear_stiefel_dims(Nx, Nv, mx, mv, metric_factors)
    return TransposeBilinearStiefelLayer{
        Int(Nx), Int(Nv), Int(mx), Int(mv), typeof(metric_factors)}(metric_factors)
end

function inverse_layer(
        l::BilinearStiefelLayer{Nx, Nv, Mx, Mv, MF}) where {Nx, Nv, Mx, Mv, MF}
    return TransposeBilinearStiefelLayer{Nx, Nv, Mx, Mv, MF}(l.metric_factors)
end

function inverse_layer(
        l::TransposeBilinearStiefelLayer{Nx, Nv, Mx, Mv, MF}) where {Nx, Nv, Mx, Mv, MF}
    return BilinearStiefelLayer{Nx, Nv, Mx, Mv, MF}(l.metric_factors)
end

function parameterlength(::BilinearStiefelLayer{Nx, Nv, Mx, Mv}) where {Nx, Nv, Mx, Mv}
    half_Nv = Nv ÷ 2
    len_x = Nx * Mx - (Mx * (Mx + 1)) ÷ 2
    len_v = half_Nv * Mv - (Mv * (Mv + 1)) ÷ 2
    return len_x + len_v
end

function parameterlength(
        l::TransposeBilinearStiefelLayer{Nx, Nv, Mx, Mv}) where {Nx, Nv, Mx, Mv}
    return parameterlength(inverse_layer(l))
end

function initialparameters(
        rng::AbstractRNG,
        init::AbstractNeuralNetworks.Initializer,
        ::BilinearStiefelLayer{Nx, Nv, Mx, Mv},
        backend::NeuralNetworkBackend,
        ::Type{T}
) where {Nx, Nv, Mx, Mv, T}
    half_Nv = Nv ÷ 2
    Φx_buf = KernelAbstractions.allocate(backend, T, Nx, Mx)
    Φv_buf = KernelAbstractions.allocate(backend, T, half_Nv, Mv)
    Φx = StiefelManifold(orthonormal_columns() do
        init(rng, Φx_buf)
        Φx_buf
    end)
    Φv = StiefelManifold(orthonormal_columns() do
        init(rng, Φv_buf)
        Φv_buf
    end)
    return (Φx = Φx, Φv = Φv)
end

function initialparameters(
        rng::AbstractRNG,
        init::AbstractNeuralNetworks.Initializer,
        l::TransposeBilinearStiefelLayer,
        backend::NeuralNetworkBackend,
        ::Type{T}
) where {T}
    return initialparameters(rng, init, inverse_layer(l), backend, T)
end

function _parity_split(F::AbstractArray{T, N}) where {T, N}
    Nv = size(F, 2)
    half_Nv = Nv ÷ 2
    inv_sqrt2 = inv(sqrt(T(2)))
    colons = ntuple(_ -> (:), Val(N - 2))
    F_top = @view F[:, 1:half_Nv, colons...]
    F_bot = @view F[:, Nv:-1:(half_Nv + 1), colons...]
    F_plus = inv_sqrt2 .* (F_top .+ F_bot)
    F_minus = inv_sqrt2 .* (F_top .- F_bot)
    return F_plus, F_minus
end

function _parity_recombine(
        F_plus::AbstractArray{T, N}, F_minus::AbstractArray{T, N}) where {T, N}
    inv_sqrt2 = inv(sqrt(T(2)))
    colons = ntuple(_ -> (:), Val(N - 2))
    F_top = inv_sqrt2 .* (F_plus .+ F_minus)
    F_bot_rev = inv_sqrt2 .* (F_plus .- F_minus)
    return cat(F_top, @view(F_bot_rev[:, end:-1:1, colons...]); dims = 2)
end

function _apply_metric_forward_2d(
        F_plus::AbstractMatrix, F_minus::AbstractMatrix, ::Nothing)
    return F_plus, F_minus
end

function _apply_metric_forward_2d(
        F_plus::AbstractMatrix, F_minus::AbstractMatrix,
        metric_factors::Tuple{AbstractMatrix, AbstractMatrix})
    Rx, Rv = metric_factors
    half_Nv = size(F_plus, 2)
    if size(Rv, 1) == half_Nv
        return Rx * F_plus * Rv', Rx * F_minus * Rv'
    else
        F_full = _parity_recombine(F_plus, F_minus)
        return _parity_split(Rx * F_full * Rv')
    end
end

function _apply_metric_forward_3d(
        F_plus::AbstractArray{T, 3}, F_minus::AbstractArray{T, 3}, ::Nothing) where {T}
    return F_plus, F_minus
end

function _apply_metric_forward_3d(
        F_plus::AbstractArray{T, 3}, F_minus::AbstractArray{T, 3},
        metric_factors::Tuple{AbstractMatrix, AbstractMatrix}) where {T}
    Rx, Rv = metric_factors
    Nx, half_Nv, B = size(F_plus)
    if size(Rv, 1) == half_Nv
        Fp_x = reshape(Rx * reshape(F_plus, Nx, half_Nv * B), Nx, half_Nv, B)
        Fm_x = reshape(Rx * reshape(F_minus, Nx, half_Nv * B), Nx, half_Nv, B)
        Fp_xv = permutedims(
            reshape(Rv * reshape(permutedims(Fp_x, (2, 1, 3)), half_Nv, Nx * B), half_Nv, Nx, B),
            (2, 1, 3))
        Fm_xv = permutedims(
            reshape(Rv * reshape(permutedims(Fm_x, (2, 1, 3)), half_Nv, Nx * B), half_Nv, Nx, B),
            (2, 1, 3))
        return Fp_xv, Fm_xv
    else
        Nv = 2 * half_Nv
        F_full = _parity_recombine(F_plus, F_minus)
        F_x = reshape(Rx * reshape(F_full, Nx, Nv * B), Nx, Nv, B)
        F_xv = permutedims(
            reshape(Rv * reshape(permutedims(F_x, (2, 1, 3)), Nv, Nx * B), Nv, Nx, B),
            (2, 1, 3))
        return _parity_split(F_xv)
    end
end

function _apply_metric_inverse_2d(
        F_plus::AbstractMatrix, F_minus::AbstractMatrix, ::Nothing)
    return _parity_recombine(F_plus, F_minus)
end

function _apply_metric_inverse_2d(
        F_plus::AbstractMatrix, F_minus::AbstractMatrix,
        metric_factors::Tuple{AbstractMatrix, AbstractMatrix})
    Rx, Rv = metric_factors
    half_Nv = size(F_plus, 2)
    if size(Rv, 1) == half_Nv
        Fp_unw = (Rx \ F_plus) / Rv'
        Fm_unw = (Rx \ F_minus) / Rv'
        return _parity_recombine(Fp_unw, Fm_unw)
    else
        F_w = _parity_recombine(F_plus, F_minus)
        return (Rx \ F_w) / Rv'
    end
end

function _apply_metric_inverse_3d(
        F_plus::AbstractArray{T, 3}, F_minus::AbstractArray{T, 3}, ::Nothing) where {T}
    return _parity_recombine(F_plus, F_minus)
end

function _apply_metric_inverse_3d(
        F_plus::AbstractArray{T, 3}, F_minus::AbstractArray{T, 3},
        metric_factors::Tuple{AbstractMatrix, AbstractMatrix}) where {T}
    Rx, Rv = metric_factors
    Nx, half_Nv, B = size(F_plus)
    if size(Rv, 1) == half_Nv
        Fp_x = reshape(Rx \ reshape(F_plus, Nx, half_Nv * B), Nx, half_Nv, B)
        Fm_x = reshape(Rx \ reshape(F_minus, Nx, half_Nv * B), Nx, half_Nv, B)
        Fp_xv = permutedims(
            reshape(Rv \ reshape(permutedims(Fp_x, (2, 1, 3)), half_Nv, Nx * B), half_Nv, Nx, B),
            (2, 1, 3))
        Fm_xv = permutedims(
            reshape(Rv \ reshape(permutedims(Fm_x, (2, 1, 3)), half_Nv, Nx * B), half_Nv, Nx, B),
            (2, 1, 3))
        return _parity_recombine(Fp_xv, Fm_xv)
    else
        Nv = 2 * half_Nv
        F_w = _parity_recombine(F_plus, F_minus)
        F_x = reshape(Rx \ reshape(F_w, Nx, Nv * B), Nx, Nv, B)
        return permutedims(
            reshape(Rv \ reshape(permutedims(F_x, (2, 1, 3)), Nv, Nx * B), Nv, Nx, B),
            (2, 1, 3))
    end
end

function _project_parity_2d(
        F_plus::AbstractMatrix, F_minus::AbstractMatrix,
        Φx::AbstractMatrix, Φv::AbstractMatrix, Mx::Int, Mv::Int)
    Q_red = (Φx' * F_plus) * Φv
    P_red = (Φx' * F_minus) * Φv
    return (q = reshape(Q_red, Mx * Mv), p = reshape(P_red, Mx * Mv))
end

function _project_parity_3d(
        F_plus::AbstractArray{T, 3}, F_minus::AbstractArray{T, 3},
        Φx::AbstractMatrix, Φv::AbstractMatrix, Nx::Int, half_Nv::Int, Mx::Int, Mv::Int) where {T}
    B = size(F_plus, 3)
    # Contract mode 1 with Φx'
    Gp = reshape(Φx' * reshape(F_plus, Nx, half_Nv * B), Mx, half_Nv, B)
    Gm = reshape(Φx' * reshape(F_minus, Nx, half_Nv * B), Mx, half_Nv, B)
    # Contract mode 2 with Φv via permutedims to keep batched BLAS gemm
    Gp_perm = reshape(permutedims(Gp, (1, 3, 2)), Mx * B, half_Nv)
    Gm_perm = reshape(permutedims(Gm, (1, 3, 2)), Mx * B, half_Nv)
    Q_perm = reshape(Gp_perm * Φv, Mx, B, Mv)
    P_perm = reshape(Gm_perm * Φv, Mx, B, Mv)
    Q_red = reshape(permutedims(Q_perm, (1, 3, 2)), Mx * Mv, B)
    P_red = reshape(permutedims(P_perm, (1, 3, 2)), Mx * Mv, B)
    return (q = Q_red, p = P_red)
end

function (l::BilinearStiefelLayer{Nx, Nv, Mx, Mv})(
        F::AbstractMatrix, ps::NamedTuple) where {Nx, Nv, Mx, Mv}
    (size(F, 1) == Nx && size(F, 2) == Nv) ||
        throw(DimensionMismatch("F must have first two dimensions ($Nx, $Nv); got size(F) = $(size(F))"))
    F_plus, F_minus = _parity_split(F)
    Fp_w, Fm_w = _apply_metric_forward_2d(F_plus, F_minus, l.metric_factors)
    return _project_parity_2d(Fp_w, Fm_w, ps.Φx, ps.Φv, Mx, Mv)
end

function (l::BilinearStiefelLayer{Nx, Nv, Mx, Mv})(
        F::AbstractArray{T, 3}, ps::NamedTuple) where {Nx, Nv, Mx, Mv, T}
    (size(F, 1) == Nx && size(F, 2) == Nv) ||
        throw(DimensionMismatch("F must have first two dimensions ($Nx, $Nv); got size(F) = $(size(F))"))
    half_Nv = Nv ÷ 2
    F_plus, F_minus = _parity_split(F)
    Fp_w, Fm_w = _apply_metric_forward_3d(F_plus, F_minus, l.metric_factors)
    return _project_parity_3d(Fp_w, Fm_w, ps.Φx, ps.Φv, Nx, half_Nv, Mx, Mv)
end

function (l::BilinearStiefelLayer{Nx, Nv, Mx, Mv})(
        x::NamedTuple{(:q, :p)}, ps::NamedTuple) where {Nx, Nv, Mx, Mv}
    size(x.q) == size(x.p) ||
        throw(DimensionMismatch("q and p must have the same size; got $(size(x.q)) and $(size(x.p))"))
    half_Nv = Nv ÷ 2
    (size(x.q, 1) == Nx && size(x.q, 2) == half_Nv) ||
        throw(DimensionMismatch("pre-split q and p must have first two dimensions ($Nx, $half_Nv); got $(size(x.q))"))
    if ndims(x.q) == 2
        Fp_w, Fm_w = _apply_metric_forward_2d(x.q, x.p, l.metric_factors)
        return _project_parity_2d(Fp_w, Fm_w, ps.Φx, ps.Φv, Mx, Mv)
    elseif ndims(x.q) == 3
        Fp_w, Fm_w = _apply_metric_forward_3d(x.q, x.p, l.metric_factors)
        return _project_parity_3d(Fp_w, Fm_w, ps.Φx, ps.Φv, Nx, half_Nv, Mx, Mv)
    else
        throw(DimensionMismatch("pre-split q and p must be 2D or 3D arrays; got ndims = $(ndims(x.q))"))
    end
end

function _reshape_latent_qp_2d(
        q::AbstractVector, p::AbstractVector, Mx::Int, Mv::Int)
    length(q) == length(p) ||
        throw(DimensionMismatch("q and p must have the same size; got $(size(q)) and $(size(p))"))
    n_lat = Mx * Mv
    length(q) == n_lat ||
        throw(DimensionMismatch("latent (q, p) must have size ($n_lat,); got $(size(q))"))
    return reshape(q, Mx, Mv), reshape(p, Mx, Mv)
end

function _reshape_latent_qp_3d(
        q::AbstractMatrix, p::AbstractMatrix, Mx::Int, Mv::Int)
    size(q) == size(p) ||
        throw(DimensionMismatch("q and p must have the same size; got $(size(q)) and $(size(p))"))
    n_lat = Mx * Mv
    size(q, 1) == n_lat ||
        throw(DimensionMismatch("latent (q, p) must have size ($n_lat, B); got $(size(q))"))
    B = size(q, 2)
    return reshape(q, Mx, Mv, B), reshape(p, Mx, Mv, B)
end

function _reshape_latent_qp_3d(
        q::AbstractArray, p::AbstractArray, Mx::Int, Mv::Int)
    throw(DimensionMismatch("latent (q, p) must have size ($(Mx * Mv),) or ($(Mx * Mv), B); got $(size(q)) and $(size(p))"))
end

function _decode_bilinear_2d(
        Q_red::AbstractMatrix, P_red::AbstractMatrix,
        Φx::AbstractMatrix, Φv::AbstractMatrix, metric_factors)
    F_plus = (Φx * Q_red) * Φv'
    F_minus = (Φx * P_red) * Φv'
    return _apply_metric_inverse_2d(F_plus, F_minus, metric_factors)
end

function _decode_bilinear_3d(
        Q_red::AbstractArray{T, 3}, P_red::AbstractArray{T, 3},
        Φx::AbstractMatrix, Φv::AbstractMatrix,
        Nx::Int, half_Nv::Int, Mx::Int, Mv::Int, metric_factors) where {T}
    B = size(Q_red, 3)
    # Contract mode 1 with Φx
    Hp = reshape(Φx * reshape(Q_red, Mx, Mv * B), Nx, Mv, B)
    Hm = reshape(Φx * reshape(P_red, Mx, Mv * B), Nx, Mv, B)
    # Contract mode 2 with Φv'
    Hp_perm = reshape(permutedims(Hp, (1, 3, 2)), Nx * B, Mv)
    Hm_perm = reshape(permutedims(Hm, (1, 3, 2)), Nx * B, Mv)
    Fp_perm = reshape(Hp_perm * Φv', Nx, B, half_Nv)
    Fm_perm = reshape(Hm_perm * Φv', Nx, B, half_Nv)
    F_plus = permutedims(Fp_perm, (1, 3, 2))
    F_minus = permutedims(Fm_perm, (1, 3, 2))
    return _apply_metric_inverse_3d(F_plus, F_minus, metric_factors)
end

function (l::TransposeBilinearStiefelLayer{Nx, Nv, Mx, Mv})(
        x::NamedTuple{(:q, :p), <:Tuple{AbstractVector, AbstractVector}},
        ps::NamedTuple) where {Nx, Nv, Mx, Mv}
    Q_red, P_red = _reshape_latent_qp_2d(x.q, x.p, Mx, Mv)
    return _decode_bilinear_2d(Q_red, P_red, ps.Φx, ps.Φv, l.metric_factors)
end

function (l::TransposeBilinearStiefelLayer{Nx, Nv, Mx, Mv})(
        x::NamedTuple{(:q, :p), <:Tuple{AbstractArray, AbstractArray}},
        ps::NamedTuple) where {Nx, Nv, Mx, Mv}
    Q_red, P_red = _reshape_latent_qp_3d(x.q, x.p, Mx, Mv)
    return _decode_bilinear_3d(
        Q_red, P_red, ps.Φx, ps.Φv, Nx, Nv ÷ 2, Mx, Mv, l.metric_factors)
end

function (l::TransposeBilinearStiefelLayer{Nx, Nv, Mx, Mv})(
        x::Tuple{AbstractMatrix, AbstractMatrix}, ps::NamedTuple) where {Nx, Nv, Mx, Mv}
    Q_red, P_red = x
    size(Q_red) == size(P_red) ||
        throw(DimensionMismatch("Q_red and P_red must have the same size; got $(size(Q_red)) and $(size(P_red))"))
    size(Q_red) == (Mx, Mv) ||
        throw(DimensionMismatch("Q_red and P_red must have size ($Mx, $Mv); got $(size(Q_red))"))
    return _decode_bilinear_2d(Q_red, P_red, ps.Φx, ps.Φv, l.metric_factors)
end

function (l::TransposeBilinearStiefelLayer{Nx, Nv, Mx, Mv})(
        x::Tuple{AbstractArray, AbstractArray}, ps::NamedTuple) where {Nx, Nv, Mx, Mv}
    return l((q = x[1], p = x[2]), ps)
end

_copy_stiefel!(dest::StiefelManifold, src::AbstractMatrix) = copyto!(dest.A, src)
_copy_stiefel!(dest::AbstractMatrix, src::AbstractMatrix) = copyto!(dest, src)

@doc raw"""
    solve!(l::BilinearStiefelLayer, ps::NamedTuple, F_snapshots::AbstractArray{T, 3})
    solve!(nn::NeuralNetwork, F_snapshots::AbstractArray{T, 3})

Initialize the Stiefel bases `ps.Φx` and `ps.Φv` of a [`BilinearStiefelLayer`](@ref) via
truncated higher-order singular value decomposition (Tucker-2 HOSVD) on a 3D snapshot array
`F_snapshots` of shape `(Nx, Nv, n_snapshots)`:

1. Splits `F_snapshots` into weighted velocity-even and velocity-odd tensors
   ``F^+, F^- \in \mathbb{R}^{N_x \times (N_v / 2) \times n_{\mathrm{snapshots}}}``.
2. Sets ``\Phi_x \in \mathrm{St}(m_x, N_x)`` to the top-``m_x`` left singular vectors of
   the spatial mode-1 unfolding ``[F_{(1)}^+\;\; F_{(1)}^-]``.
3. Sets ``\Phi_v \in \mathrm{St}(m_v, N_v / 2)`` to the top-``m_v`` left singular vectors of
   the parity-velocity mode-2 unfolding ``[F_{(2)}^+\;\; F_{(2)}^-]``.
"""
function solve!(
        l::BilinearStiefelLayer{Nx, Nv, Mx, Mv},
        ps::NamedTuple,
        F_snapshots::AbstractArray{T, 3}
) where {Nx, Nv, Mx, Mv, T}
    half_Nv = Nv ÷ 2
    n_snaps = size(F_snapshots, 3)
    (size(F_snapshots, 1) == Nx && size(F_snapshots, 2) == Nv && n_snaps >= 1) ||
        throw(DimensionMismatch("F_snapshots must have first two dimensions ($Nx, $Nv) and at least 1 snapshot; got $(size(F_snapshots))"))
    F_plus, F_minus = _parity_split(F_snapshots)
    Fp_w, Fm_w = _apply_metric_forward_3d(F_plus, F_minus, l.metric_factors)

    # Spatial mode-1 unfolding: Nx × (Nv * n_snaps)
    M_x = hcat(reshape(Fp_w, Nx, half_Nv * n_snaps), reshape(Fm_w, Nx, half_Nv * n_snaps))
    svd_x = svd(M_x; full = true)
    _copy_stiefel!(ps.Φx, @view(svd_x.U[:, 1:Mx]))

    # Parity-velocity mode-2 unfolding: (Nv ÷ 2) × (2 * Nx * n_snaps)
    M_v = hcat(
        reshape(permutedims(Fp_w, (2, 1, 3)), half_Nv, Nx * n_snaps),
        reshape(permutedims(Fm_w, (2, 1, 3)), half_Nv, Nx * n_snaps))
    svd_v = svd(M_v; full = true)
    _copy_stiefel!(ps.Φv, @view(svd_v.U[:, 1:Mv]))
    return ps
end

function _solve_bilinear_stiefel!(
        nn::NeuralNetwork, F_snapshots::AbstractArray{T, 3}) where {T}
    l_enc = nn.model.layers[1]
    ps_enc = nn.params.L1
    solve!(l_enc, ps_enc, F_snapshots)
    n_layers = length(nn.model.layers)
    if n_layers > 1 && nn.model.layers[end] isa TransposeBilinearStiefelLayer
        ps_dec = nn.params[n_layers]
        if haskey(ps_dec, :Φx) && haskey(ps_dec, :Φv) &&
           size(ps_dec.Φx) == size(ps_enc.Φx) && size(ps_dec.Φv) == size(ps_enc.Φv)
            _copy_stiefel!(ps_dec.Φx, ps_enc.Φx)
            _copy_stiefel!(ps_dec.Φv, ps_enc.Φv)
        end
    end
    return nn
end

function solve!(
        nn::NeuralNetwork{<:Any, <:Chain{<:Tuple{BilinearStiefelLayer, Vararg}}},
        F_snapshots::AbstractArray{T, 3}
) where {T}
    return _solve_bilinear_stiefel!(nn, F_snapshots)
end

function solve!(
        nn::NeuralNetwork{<:PSDArch, <:Chain{<:Tuple{BilinearStiefelLayer, Vararg}}},
        F_snapshots::AbstractArray{T, 3}
) where {T}
    return _solve_bilinear_stiefel!(nn, F_snapshots)
end
