@doc raw"""
    LowRankStiefelLayer(Nx, Nv, mx, mv; r_out = mx * mv, metric_factors = nothing)

Construct a gauge-invariant dynamical low-rank approximation (DLRA) Stiefel encoder layer
that maps factored rank-``r_{\mathrm{in}}`` states ``(X = X, S = S, V = V)`` with
``X \in \mathbb{R}^{N_x \times r_{\mathrm{in}}}``,
``S \in \mathbb{R}^{r_{\mathrm{in}} \times r_{\mathrm{in}}}``, and
``V \in \mathbb{R}^{N_v \times r_{\mathrm{in}}}`` (or 3D batched arrays) to canonical
coordinates ``(q, p)`` of half-dimension ``n = m_x m_v`` in
``O(r_{\mathrm{in}}(m_x N_x + m_v N_v))`` operations without assembling the dense
``N_x \times N_v`` matrix ``X S V^T``.

Splitting ``V`` into velocity-even and velocity-odd half-grid factors
``V^+, V^- \in \mathbb{R}^{(N_v / 2) \times r_{\mathrm{in}}}`` and contracting with
``\Phi_x \in \mathrm{St}(m_x, N_x)`` and ``\Phi_v \in \mathrm{St}(m_v, N_v / 2)`` gives

```math
Q_{\mathrm{red}} = (\Phi_x^T X)\, S\, \bigl((V^+)^T \Phi_v\bigr), \qquad
P_{\mathrm{red}} = (\Phi_x^T X)\, S\, \bigl((V^-)^T \Phi_v\bigr),
```

which produces the exact same canonical coordinates `(q = vec(Q_red), p = vec(P_red))` as
[`BilinearStiefelLayer`](@ref) applied to ``F = X S V^T`` and is strictly invariant under
orthogonal gauge transformations ``(X, S, V) \mapsto (X O_x, O_x^T S O_v, V O_v)`` for all
``O_x, O_v \in O(r_{\mathrm{in}})``.

See also [`DecodeLowRankStiefelLayer`](@ref), [`BilinearStiefelLayer`](@ref),
[`FastDLRATraceLoss`](@ref), and [`inverse_layer`](@ref).
"""
struct LowRankStiefelLayer{Nx, Nv, Mx, Mv, ROut, MF} <: AbstractExplicitLayer{Nx, Mx}
    metric_factors::MF
end

@doc raw"""
    DecodeLowRankStiefelLayer(Nx, Nv, mx, mv; r_out = mx * mv, metric_factors = nothing)

Construct the Stiefel-preserving low-rank SVD decoder layer corresponding to
[`LowRankStiefelLayer`](@ref).

Given canonical coordinates `(q = q, p = p)` (or `(Q_red, P_red)`), the layer forms the
``m_x \times 2 m_v`` matrix ``[Q_{\mathrm{red}}\;\; P_{\mathrm{red}}]``, computes its small
singular value decomposition ``U_S \Sigma_S W_S^T`` truncated to rank
``r_{\mathrm{eff}} = \min(r_{\mathrm{out}}, m_x, 2 m_v)``, lifts ``\tilde{X} = \Phi_x U_S``
and ``\tilde{V}`` from ``W_S`` and ``\Phi_v`` (applying ``(R_x^{-1}, R_v^{-1})`` when
`metric_factors` is supplied), and QR re-orthonormalizes ``\tilde{X} = \hat{X} R_X`` and
``\tilde{V} = \hat{V} R_V`` with ``\hat{S} = R_X \Sigma_S R_V^T`` so that the returned
factors `(X = X_hat, S = S_hat, V = V_hat)` satisfy
``\hat{X}^T \hat{X} = I_{r_{\mathrm{eff}}}`` and
``\hat{V}^T \hat{V} = I_{r_{\mathrm{eff}}}``.

See also [`LowRankStiefelLayer`](@ref), [`TransposeBilinearStiefelLayer`](@ref), and
[`inverse_layer`](@ref).
"""
struct DecodeLowRankStiefelLayer{Nx, Nv, Mx, Mv, ROut, MF} <: AbstractExplicitLayer{Mx, Nx}
    metric_factors::MF
end

function LowRankStiefelLayer(
        Nx::Integer, Nv::Integer, mx::Integer, mv::Integer;
        r_out::Integer = mx * mv, metric_factors = nothing)
    _validate_bilinear_stiefel_dims(Nx, Nv, mx, mv, metric_factors)
    r_out >= 1 || throw(ArgumentError("r_out must be ≥ 1; got r_out = $r_out"))
    r_eff = min(Int(r_out), Int(mx), 2 * Int(mv))
    return LowRankStiefelLayer{
        Int(Nx), Int(Nv), Int(mx), Int(mv), r_eff, typeof(metric_factors)}(metric_factors)
end

function DecodeLowRankStiefelLayer(
        Nx::Integer, Nv::Integer, mx::Integer, mv::Integer;
        r_out::Integer = mx * mv, metric_factors = nothing)
    _validate_bilinear_stiefel_dims(Nx, Nv, mx, mv, metric_factors)
    r_out >= 1 || throw(ArgumentError("r_out must be ≥ 1; got r_out = $r_out"))
    r_eff = min(Int(r_out), Int(mx), 2 * Int(mv))
    return DecodeLowRankStiefelLayer{
        Int(Nx), Int(Nv), Int(mx), Int(mv), r_eff, typeof(metric_factors)}(metric_factors)
end

function inverse_layer(
        l::LowRankStiefelLayer{Nx, Nv, Mx, Mv, ROut, MF}
) where {Nx, Nv, Mx, Mv, ROut, MF}
    return DecodeLowRankStiefelLayer{Nx, Nv, Mx, Mv, ROut, MF}(l.metric_factors)
end

function inverse_layer(
        l::DecodeLowRankStiefelLayer{
        Nx, Nv, Mx, Mv, ROut, MF}
) where {Nx, Nv, Mx, Mv, ROut, MF}
    return LowRankStiefelLayer{Nx, Nv, Mx, Mv, ROut, MF}(l.metric_factors)
end

function parameterlength(
        ::LowRankStiefelLayer{Nx, Nv, Mx, Mv}) where {Nx, Nv, Mx, Mv}
    return parameterlength(BilinearStiefelLayer{Nx, Nv, Mx, Mv, Nothing}(nothing))
end

function parameterlength(
        l::DecodeLowRankStiefelLayer{Nx, Nv, Mx, Mv}) where {Nx, Nv, Mx, Mv}
    return parameterlength(inverse_layer(l))
end

function initialparameters(
        rng::AbstractRNG,
        init::AbstractNeuralNetworks.Initializer,
        ::LowRankStiefelLayer{Nx, Nv, Mx, Mv},
        backend::NeuralNetworkBackend,
        ::Type{T}
) where {Nx, Nv, Mx, Mv, T}
    return initialparameters(
        rng, init, BilinearStiefelLayer{Nx, Nv, Mx, Mv, Nothing}(nothing), backend, T)
end

function initialparameters(
        rng::AbstractRNG,
        init::AbstractNeuralNetworks.Initializer,
        l::DecodeLowRankStiefelLayer,
        backend::NeuralNetworkBackend,
        ::Type{T}
) where {T}
    return initialparameters(rng, init, inverse_layer(l), backend, T)
end

function _validate_dlra_factors(
        X::AbstractMatrix, S::AbstractMatrix, V::AbstractMatrix, Nx::Int, Nv::Int)
    size(X, 1) == Nx ||
        throw(DimensionMismatch("X must have $Nx rows; got size(X) = $(size(X))"))
    r_in = size(X, 2)
    r_in >= 1 ||
        throw(DimensionMismatch("DLRA rank r_in must be ≥ 1; got r_in = $r_in"))
    size(S) == (r_in, r_in) ||
        throw(DimensionMismatch("S must have size ($r_in, $r_in); got size(S) = $(size(S))"))
    size(V) == (Nv, r_in) ||
        throw(DimensionMismatch("V must have size ($Nv, $r_in); got size(V) = $(size(V))"))
    return r_in
end

function _validate_dlra_factors(
        X::AbstractArray{<:Any, 3}, S::AbstractArray{<:Any, 3}, V::AbstractArray{<:Any, 3},
        Nx::Int, Nv::Int)
    size(X, 1) == Nx ||
        throw(DimensionMismatch("X must have $Nx rows; got size(X) = $(size(X))"))
    r_in = size(X, 2)
    r_in >= 1 ||
        throw(DimensionMismatch("DLRA rank r_in must be ≥ 1; got r_in = $r_in"))
    B = size(X, 3)
    size(S) == (r_in, r_in, B) ||
        throw(DimensionMismatch("S must have size ($r_in, $r_in, $B); got size(S) = $(size(S))"))
    size(V) == (Nv, r_in, B) ||
        throw(DimensionMismatch("V must have size ($Nv, $r_in, $B); got size(V) = $(size(V))"))
    return r_in, B
end

function _parity_split_v(V::AbstractMatrix{T}) where {T}
    Nv = size(V, 1)
    half_Nv = Nv ÷ 2
    inv_sqrt2 = inv(sqrt(T(2)))
    V_top = @view V[1:half_Nv, :]
    V_bot = @view V[Nv:-1:(half_Nv + 1), :]
    return inv_sqrt2 .* (V_top .+ V_bot), inv_sqrt2 .* (V_top .- V_bot)
end

function _encode_dlra_2d(
        X::AbstractMatrix, S::AbstractMatrix, V::AbstractMatrix,
        Φx::AbstractMatrix, Φv::AbstractMatrix,
        Mx::Int, Mv::Int, metric_factors)
    if metric_factors === nothing
        X_eff = X
        V_plus, V_minus = _parity_split_v(V)
    else
        Rx, Rv = metric_factors
        X_eff = Rx * X
        half_Nv = size(V, 1) ÷ 2
        if size(Rv, 1) == half_Nv
            Vp0, Vm0 = _parity_split_v(V)
            V_plus = Rv * Vp0
            V_minus = Rv * Vm0
        else
            V_plus, V_minus = _parity_split_v(Rv * V)
        end
    end
    Hx = Φx' * X_eff
    Hv_plus = V_plus' * Φv
    Hv_minus = V_minus' * Φv
    HxS = Hx * S
    Q_red = HxS * Hv_plus
    P_red = HxS * Hv_minus
    return (q = reshape(Q_red, Mx * Mv), p = reshape(P_red, Mx * Mv))
end

function (l::LowRankStiefelLayer{Nx, Nv, Mx, Mv})(
        x::NamedTuple{(:X, :S, :V)}, ps::NamedTuple) where {Nx, Nv, Mx, Mv}
    if ndims(x.X) == 2
        _validate_dlra_factors(x.X, x.S, x.V, Nx, Nv)
        return _encode_dlra_2d(
            x.X, x.S, x.V, ps.Φx, ps.Φv, Mx, Mv, l.metric_factors)
    elseif ndims(x.X) == 3
        _, B = _validate_dlra_factors(x.X, x.S, x.V, Nx, Nv)
        n_lat = Mx * Mv
        T = promote_type(
            eltype(x.X), eltype(x.S), eltype(x.V), eltype(ps.Φx), eltype(ps.Φv))
        q_out = Matrix{T}(undef, n_lat, B)
        p_out = Matrix{T}(undef, n_lat, B)
        for b in 1:B
            qp_b = _encode_dlra_2d(
                @view(x.X[:, :, b]), @view(x.S[:, :, b]), @view(x.V[:, :, b]),
                ps.Φx, ps.Φv, Mx, Mv, l.metric_factors)
            q_out[:, b] = qp_b.q
            p_out[:, b] = qp_b.p
        end
        return (q = q_out, p = p_out)
    else
        throw(DimensionMismatch("DLRA factors must be 2D or 3D arrays; got ndims = $(ndims(x.X))"))
    end
end

function _decode_dlra_2d(
        Q_red::AbstractMatrix, P_red::AbstractMatrix,
        Φx::AbstractMatrix, Φv::AbstractMatrix,
        Mv::Int, ROut::Int, metric_factors)
    T = promote_type(eltype(Q_red), eltype(P_red), eltype(Φx), eltype(Φv))
    M_cat = hcat(Q_red, P_red)
    F_svd = svd(M_cat)
    r = min(ROut, size(F_svd.U, 2), size(F_svd.V, 2))
    U_S = @view F_svd.U[:, 1:r]
    s_vec = @view F_svd.S[1:r]
    W_S = @view F_svd.V[:, 1:r]
    W_plus = @view W_S[1:Mv, :]
    W_minus = @view W_S[(Mv + 1):(2 * Mv), :]

    X_tilde = Φx * U_S
    V_plus = Φv * W_plus
    V_minus = Φv * W_minus

    if metric_factors !== nothing
        Rx, Rv = metric_factors
        X_tilde = Rx \ X_tilde
        if size(Rv, 1) == size(V_plus, 1)
            V_plus = Rv \ V_plus
            V_minus = Rv \ V_minus
        end
    end

    inv_sqrt2 = inv(sqrt(T(2)))
    V_top = inv_sqrt2 .* (V_plus .+ V_minus)
    V_bot_rev = inv_sqrt2 .* (V_plus .- V_minus)
    V_tilde = vcat(V_top, @view(V_bot_rev[end:-1:1, :]))

    if metric_factors !== nothing && size(metric_factors[2], 1) == size(V_tilde, 1)
        V_tilde = metric_factors[2] \ V_tilde
    end

    # Unconditional QR re-orthonormalization so X_hat ∈ St(r, Nx) and V_hat ∈ St(r, Nv)
    qr_X = qr(X_tilde)
    X_hat = Matrix{T}(qr_X.Q)[:, 1:r]
    R_X = Matrix{T}(qr_X.R)[1:r, 1:r]

    qr_V = qr(V_tilde)
    V_hat = Matrix{T}(qr_V.Q)[:, 1:r]
    R_V = Matrix{T}(qr_V.R)[1:r, 1:r]

    S_hat = R_X * Diagonal(s_vec) * R_V'
    return (X = X_hat, S = S_hat, V = V_hat)
end

function (l::DecodeLowRankStiefelLayer{Nx, Nv, Mx, Mv, ROut})(
        x::NamedTuple{(:q, :p), <:Tuple{AbstractVector, AbstractVector}},
        ps::NamedTuple) where {Nx, Nv, Mx, Mv, ROut}
    Q_red, P_red = _reshape_latent_qp_2d(x.q, x.p, Mx, Mv)
    return _decode_dlra_2d(
        Q_red, P_red, ps.Φx, ps.Φv, Mv, ROut, l.metric_factors)
end

function (l::DecodeLowRankStiefelLayer{Nx, Nv, Mx, Mv, ROut})(
        x::NamedTuple{(:q, :p), <:Tuple{AbstractArray, AbstractArray}},
        ps::NamedTuple) where {Nx, Nv, Mx, Mv, ROut}
    Q_red, P_red = _reshape_latent_qp_3d(x.q, x.p, Mx, Mv)
    B = size(Q_red, 3)
    r = min(ROut, Mx, 2 * Mv)
    T = promote_type(eltype(Q_red), eltype(P_red), eltype(ps.Φx), eltype(ps.Φv))
    X_out = Array{T, 3}(undef, Nx, r, B)
    S_out = Array{T, 3}(undef, r, r, B)
    V_out = Array{T, 3}(undef, Nv, r, B)
    for b in 1:B
        dec_b = _decode_dlra_2d(
            @view(Q_red[:, :, b]), @view(P_red[:, :, b]),
            ps.Φx, ps.Φv, Mv, ROut, l.metric_factors)
        X_out[:, :, b] = dec_b.X
        S_out[:, :, b] = dec_b.S
        V_out[:, :, b] = dec_b.V
    end
    return (X = X_out, S = S_out, V = V_out)
end

function (l::DecodeLowRankStiefelLayer{Nx, Nv, Mx, Mv, ROut})(
        x::Tuple{AbstractMatrix, AbstractMatrix}, ps::NamedTuple) where {
        Nx, Nv, Mx, Mv, ROut}
    Q_red, P_red = x
    size(Q_red) == size(P_red) ||
        throw(DimensionMismatch("Q_red and P_red must have the same size; got $(size(Q_red)) and $(size(P_red))"))
    size(Q_red) == (Mx, Mv) ||
        throw(DimensionMismatch("Q_red and P_red must have size ($Mx, $Mv); got $(size(Q_red))"))
    return _decode_dlra_2d(
        Q_red, P_red, ps.Φx, ps.Φv, Mv, ROut, l.metric_factors)
end

function (l::DecodeLowRankStiefelLayer{Nx, Nv, Mx, Mv, ROut})(
        x::Tuple{AbstractArray, AbstractArray}, ps::NamedTuple) where {Nx, Nv, Mx, Mv, ROut}
    return l((q = x[1], p = x[2]), ps)
end
