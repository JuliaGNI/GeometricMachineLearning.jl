_state_eltype(::AbstractArray{T}) where {T} = T
_state_eltype(nt::NamedTuple{(:q, :p)}) = promote_type(eltype(nt.q), eltype(nt.p))
function _state_eltype(nt::NamedTuple{(:X, :S, :V)})
    promote_type(eltype(nt.X), eltype(nt.S), eltype(nt.V))
end

_frob_sq_norm(x::AbstractArray) = sum(abs2, x)
_frob_sq_norm(x::NamedTuple{(:q, :p)}) = sum(abs2, x.q) + sum(abs2, x.p)
function _frob_sq_norm(x::NamedTuple{(:X, :S, :V)})
    G_X = transpose(x.X) * x.X
    G_V = transpose(x.V) * x.V
    S_GV = x.S * G_V
    T = _state_eltype(x)
    max(zero(T), tr(G_X * S_GV * transpose(x.S)))
end
# A batch of DLRA states, one per slice along axis 3: the sum of the slices' squared norms.
function _frob_sq_norm(x::NamedTuple{(:X, :S, :V), <:NTuple{3, AbstractArray{<:Any, 3}}})
    T = _state_eltype(x)
    X, S, V = T.(x.X), T.(x.S), T.(x.V)
    G_X = tensor_transpose_tensor_mul(X, X)
    G_V = tensor_transpose_tensor_mul(V, V)
    M = tensor_tensor_mul(tensor_transpose_tensor_mul(S, G_X), S)
    sum(max.(zero(T), sum(M .* permutedims(G_V, (2, 1, 3)); dims = (1, 2))))
end

_frob_sq_diff(x̂::AbstractArray, x::AbstractArray) = sum(abs2, x̂ .- x)
function _frob_sq_diff(x̂::NamedTuple{(:q, :p)}, x::NamedTuple{(:q, :p)})
    sum(abs2, x̂.q .- x.q) + sum(abs2, x̂.p .- x.p)
end
function _frob_sq_diff(x̂::AbstractVector, x::NamedTuple{(:q, :p)})
    sum(abs2, x̂ .- vcat(vec(x.q), vec(x.p)))
end
function _frob_sq_diff(x̂::NamedTuple{(:q, :p)}, x::AbstractVector)
    _frob_sq_diff(x, x̂)
end
function _frob_sq_diff(x̂::AbstractMatrix, x::NamedTuple{(:q, :p)})
    if ndims(x.q) <= 2
        sum(abs2, vec(x̂) .- vcat(vec(x.q), vec(x.p)))
    else
        N = size(x.q, 1) * size(x.q, 2)
        B = size(x.q, 3)
        sum(abs2, x̂ .- vcat(reshape(x.q, N, B), reshape(x.p, N, B)))
    end
end
function _frob_sq_diff(x̂::NamedTuple{(:q, :p)}, x::AbstractMatrix)
    _frob_sq_diff(x, x̂)
end
function _frob_sq_diff(x̂::NamedTuple{(:X, :S, :V)}, x::NamedTuple{(:X, :S, :V)})
    FastDLRATraceLoss()(x̂, x)
end

@doc raw"""
    CharbonnierRelativeLoss(; ε = 1e-3, λ_lat = 1e-2)

Smooth Charbonnier relative reconstruction and latent consistency loss:

```math
\mathcal{L}_{\mathrm{Charb}}(\hat{x}, x) =
\frac{\sqrt{\|\hat{x} - x\|_F^2 + \varepsilon^2} - \varepsilon}{\sqrt{\|x\|_F^2 + \varepsilon^2}}
+ \lambda_{\mathrm{lat}}\,\frac{\|\mathrm{encode}(nn, m, \hat{x}) - \mathrm{encode}(nn, m, x)\|_F^2}{\|x\|_F^2 + \varepsilon^2},
```

where ``\varepsilon > 0`` (default `1e-3`) regularizes the Frobenius norm at ``\hat{x} = x`` and ``x = 0``, and ``\lambda_{\mathrm{lat}} \ge 0`` (default `1e-2`) weights the latent re-encoding consistency penalty.

When ``x = 0`` and ``\hat{x} = 0``, the loss returns exact `zero(T)` (not `NaN`), and its gradient at ``\hat{x} = x`` is finite and zero (`0 / ε = 0`).
"""
struct CharbonnierRelativeLoss{T <: Real} <: NetworkLoss
    ε::T
    λ_lat::T

    function CharbonnierRelativeLoss(; ε::Real = 1e-3, λ_lat::Real = 1e-2)
        (ε > 0 && λ_lat >= 0) || throw(ArgumentError(
            "CharbonnierRelativeLoss requires ε > 0 and λ_lat >= 0, got ε = $ε, λ_lat = $λ_lat."))
        T = promote_type(typeof(float(ε)), typeof(float(λ_lat)))
        new{T}(T(ε), T(λ_lat))
    end
end

function _charbonnier_eval(
        loss::CharbonnierRelativeLoss, x̂, x, z_hat::AbstractArray, z::AbstractArray)
    T = promote_type(_state_eltype(x̂), _state_eltype(x), eltype(z_hat), eltype(z))
    eps_T = T(loss.ε)
    lam_T = T(loss.λ_lat)
    eps_sq = eps_T * eps_T
    sq_diff = _frob_sq_diff(x̂, x)
    denom_sq = _frob_sq_norm(x) + eps_sq
    num_rec = sqrt(sq_diff + eps_sq) - eps_T
    term_rec = num_rec / sqrt(denom_sq)
    term_lat = lam_T * _frob_sq_diff(z_hat, z) / denom_sq
    (term_rec + term_lat)::T
end

Base.@constprop :aggressive function (loss::CharbonnierRelativeLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple, modality::Symbol, x̂, x)
    T = promote_type(_state_eltype(x̂), _state_eltype(x), _leaf_eltype(ps.core))
    z = encode(model, ps, modality, x)
    z_hat = encode(model, ps, modality, x̂)
    _charbonnier_eval(loss, x̂, x, z_hat, z)::T
end

Base.@constprop :aggressive function (loss::CharbonnierRelativeLoss)(
        model::MultiModalSymplecticLayer, ps::NetworkParameters, modality::Symbol, x̂, x)
    loss(model, params(ps), modality, x̂, x)
end

Base.@constprop :aggressive function (loss::CharbonnierRelativeLoss)(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, modality::Symbol, x̂, x)
    loss(nn.model, params(nn), modality, x̂, x)
end

Base.@constprop :aggressive function (loss::CharbonnierRelativeLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple, modality::Symbol, x)
    T = promote_type(_state_eltype(x), _leaf_eltype(ps.core))
    z = encode(model, ps, modality, x)
    x̂ = decode(model, ps, modality, z)
    z_hat = encode(model, ps, modality, x̂)
    _charbonnier_eval(loss, x̂, x, z_hat, z)::T
end

Base.@constprop :aggressive function (loss::CharbonnierRelativeLoss)(
        model::MultiModalSymplecticLayer, ps::NetworkParameters, modality::Symbol, x)
    loss(model, params(ps), modality, x)
end

Base.@constprop :aggressive function (loss::CharbonnierRelativeLoss)(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, modality::Symbol, x)
    loss(nn.model, params(nn), modality, x)
end

Base.@constprop :aggressive function (loss::CharbonnierRelativeLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple, x)
    mod = first(keys(model.frontends))
    loss(model, ps, mod, x)
end

Base.@constprop :aggressive function (loss::CharbonnierRelativeLoss)(
        model::MultiModalSymplecticLayer, ps::NetworkParameters, x)
    loss(model, params(ps), x)
end

Base.@constprop :aggressive function (loss::CharbonnierRelativeLoss)(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, x)
    loss(nn.model, params(nn), x)
end

@doc raw"""
    IsometricRolloutLoss(integrator; α_lat = 1.0, α_rec = 1.0, α_H = 0.1, H = nothing)

Multi-step latent rollout, decoded trajectory reconstruction, and latent Hamiltonian conservation loss:

```math
\mathcal{L}_{\mathrm{roll}} =
\frac{1}{K - 1}\sum_{k=2}^{K}
\left(
\alpha_{\mathrm{lat}}\,\|\hat{z}_k - z_k\|_F^2
+ \alpha_{\mathrm{rec}}\,\|\mathrm{decode}(nn, m, \hat{z}_k) - x_k\|_F^2
+ \alpha_H\,\bigl|H(\hat{z}_k) - H(\hat{z}_1)\bigr|^2
\right),
```

where ``z_k = \mathrm{encode}(nn, m, x_k)``, ``\hat{z}_1 = z_1``, ``\hat{z}_{k+1} = \mathrm{integrator}(\hat{z}_k)`` for ``k = 1, \dots, K - 1``, and `H` is a callable latent Hamiltonian (default `nothing`, which uses the canonical quadratic energy ``\frac{1}{2}\|\hat{z}\|_2^2``).
The weights default to `α_lat = 1.0`, `α_rec = 1.0`, and `α_H = 0.1`.
"""
struct IsometricRolloutLoss{IT, HT, T <: Real} <: NetworkLoss
    integrator::IT
    H::HT
    α_lat::T
    α_rec::T
    α_H::T

    function IsometricRolloutLoss(
            integrator;
            α_lat::Real = 1.0,
            α_rec::Real = 1.0,
            α_H::Real = 0.1,
            H = nothing)
        (α_lat >= 0 && α_rec >= 0 && α_H >= 0) || throw(ArgumentError(
            "IsometricRolloutLoss requires non-negative weights α_lat, α_rec, α_H, got α_lat = $α_lat, α_rec = $α_rec, α_H = $α_H."))
        T = promote_type(typeof(float(α_lat)), typeof(float(α_rec)), typeof(float(α_H)))
        new{typeof(integrator), typeof(H), T}(
            integrator, H, T(α_lat), T(α_rec), T(α_H))
    end
end

_eval_latent_H(::Nothing, z::AbstractArray{T}) where {T} = T(0.5) * sum(abs2, z)
_eval_latent_H(H_fn, z::AbstractArray) = H_fn(z)

function _check_traj_len(K::Int)
    K >= 2 || throw(ArgumentError(
        "IsometricRolloutLoss requires at least 2 trajectory steps, got $K."))
    nothing
end

ChainRulesCore.@non_differentiable _check_traj_len(::Any)

Base.@constprop :aggressive function (loss::IsometricRolloutLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple,
        modality::Symbol, xs::AbstractVector)
    K = length(xs)
    _check_traj_len(K)
    T = promote_type(_state_eltype(xs[1]), _leaf_eltype(ps.core))
    z1 = encode(model, ps, modality, xs[1])
    H1 = _eval_latent_H(loss.H, z1)
    z_curr = z1
    err_lat = zero(T)
    err_rec = zero(T)
    err_H = zero(T)
    for k in 2:K
        z_k = encode(model, ps, modality, xs[k])
        z_curr = loss.integrator(z_curr)
        err_lat = err_lat + _frob_sq_diff(z_curr, z_k)
        x_rec_k = decode(model, ps, modality, z_curr)
        err_rec = err_rec + _frob_sq_diff(x_rec_k, xs[k])
        Hk = _eval_latent_H(loss.H, z_curr)
        err_H = err_H + abs2(Hk - H1)
    end
    inv_steps = inv(T(K - 1))
    res = T(loss.α_lat) * err_lat * inv_steps +
          T(loss.α_rec) * err_rec * inv_steps +
          T(loss.α_H) * err_H * inv_steps
    res::T
end

Base.@constprop :aggressive function (loss::IsometricRolloutLoss)(
        model::MultiModalSymplecticLayer, ps::NetworkParameters,
        modality::Symbol, xs::AbstractVector)
    loss(model, params(ps), modality, xs)
end

Base.@constprop :aggressive function (loss::IsometricRolloutLoss)(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder},
        modality::Symbol, xs::AbstractVector)
    loss(nn.model, params(nn), modality, xs)
end

Base.@constprop :aggressive function (loss::IsometricRolloutLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple, xs::AbstractVector)
    mod = first(keys(model.frontends))
    loss(model, ps, mod, xs)
end

Base.@constprop :aggressive function (loss::IsometricRolloutLoss)(
        model::MultiModalSymplecticLayer, ps::NetworkParameters, xs::AbstractVector)
    loss(model, params(ps), xs)
end

Base.@constprop :aggressive function (loss::IsometricRolloutLoss)(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, xs::AbstractVector)
    loss(nn.model, params(nn), xs)
end

@doc raw"""
    CrossModalCycleLoss(mod_a::Symbol, mod_b::Symbol; β_align = 1.0, β_cycle = 0.1)

Cross-modal paired latent alignment and cycle consistency loss between modalities `mod_a` and `mod_b`:

```math
\mathcal{L}_{\mathrm{cross}}(x_a, x_b) =
\beta_{\mathrm{align}}\,\|z_a - z_b\|_F^2
+ \beta_{\mathrm{cycle}}\,\|\mathrm{encode}(nn, b, \mathrm{decode}(nn, b, z_a)) - z_a\|_F^2,
```

where ``z_a = \mathrm{encode}(nn, a, x_a)`` and ``z_b = \mathrm{encode}(nn, b, x_b)``.
The weights default to `β_align = 1.0` and `β_cycle = 0.1`.
"""
struct CrossModalCycleLoss{T <: Real} <: NetworkLoss
    mod_a::Symbol
    mod_b::Symbol
    β_align::T
    β_cycle::T

    function CrossModalCycleLoss(
            mod_a::Symbol, mod_b::Symbol;
            β_align::Real = 1.0, β_cycle::Real = 0.1)
        (β_align >= 0 && β_cycle >= 0) || throw(ArgumentError(
            "CrossModalCycleLoss requires non-negative weights β_align and β_cycle, got β_align = $β_align, β_cycle = $β_cycle."))
        T = promote_type(typeof(float(β_align)), typeof(float(β_cycle)))
        new{T}(mod_a, mod_b, T(β_align), T(β_cycle))
    end
end

Base.@constprop :aggressive function _cross_modal_eval(
        loss::CrossModalCycleLoss, model::MultiModalSymplecticLayer,
        ps::NamedTuple, x_a, x_b)
    T = promote_type(_state_eltype(x_a), _state_eltype(x_b), _leaf_eltype(ps.core))
    z_a = encode(model, ps, loss.mod_a, x_a)
    z_b = encode(model, ps, loss.mod_b, x_b)
    x_b_rec = decode(model, ps, loss.mod_b, z_a)
    z_a_cyc = encode(model, ps, loss.mod_b, x_b_rec)
    res = T(loss.β_align) * _frob_sq_diff(z_a, z_b) +
          T(loss.β_cycle) * _frob_sq_diff(z_a_cyc, z_a)
    res::T
end

Base.@constprop :aggressive function (loss::CrossModalCycleLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple,
        x_a::ArrayOrNamedTuple, x_b::ArrayOrNamedTuple)
    _cross_modal_eval(loss, model, ps, x_a, x_b)
end

Base.@constprop :aggressive function (loss::CrossModalCycleLoss)(
        model::MultiModalSymplecticLayer, ps::NetworkParameters,
        x_a::ArrayOrNamedTuple, x_b::ArrayOrNamedTuple)
    _cross_modal_eval(loss, model, params(ps), x_a, x_b)
end

Base.@constprop :aggressive function (loss::CrossModalCycleLoss)(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder},
        x_a::ArrayOrNamedTuple, x_b::ArrayOrNamedTuple)
    loss(nn.model, params(nn), x_a, x_b)
end

Base.@constprop :aggressive function (loss::CrossModalCycleLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple, pair::Tuple{Any, Any})
    _cross_modal_eval(loss, model, ps, pair[1], pair[2])
end

Base.@constprop :aggressive function (loss::CrossModalCycleLoss)(
        model::MultiModalSymplecticLayer, ps::NetworkParameters, pair::Tuple{Any, Any})
    _cross_modal_eval(loss, model, params(ps), pair[1], pair[2])
end

Base.@constprop :aggressive function (loss::CrossModalCycleLoss)(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, pair::Tuple{Any, Any})
    loss(nn.model, params(nn), pair)
end
