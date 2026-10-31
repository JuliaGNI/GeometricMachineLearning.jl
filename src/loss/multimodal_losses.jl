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
    X = eltype(x.X) === T ? x.X : T.(x.X)
    S = eltype(x.S) === T ? x.S : T.(x.S)
    V = eltype(x.V) === T ? x.V : T.(x.V)
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

# The data that a decoded state of a frontend is compared with, in the form that `decode`
# returns for it. `decode` through a `BilinearStiefelLayer` returns the full field, so a pre-split
# pair `(q = F⁺, p = F⁻)` is recombined into that field; every other input is compared as it is.
# The two halves of a pre-split pair must have one element type.
function _check_split_eltypes(q, p)
    eltype(q) === eltype(p) || throw(ArgumentError(
        "A pre-split pair (q = F⁺, p = F⁻) needs one element type, got q of element type $(eltype(q)) and p of element type $(eltype(p))."))
    nothing
end

ChainRulesCore.@non_differentiable _check_split_eltypes(::Any, ::Any)

function _reconstruction_target(::BilinearStiefelLayer, x::NamedTuple{(:q, :p)})
    _check_split_eltypes(x.q, x.p)
    _parity_recombine(x.q, x.p)
end
_reconstruction_target(::Any, x) = x
function _reconstruction_target(model::MultiModalSymplecticLayer, modality::Symbol, x)
    _reconstruction_target(getfield(model.frontends, modality), x)
end

@doc raw"""
    CharbonnierRelativeLoss(; ε = 1e-3, λ_lat = 1e-2)

Smooth Charbonnier relative reconstruction and latent consistency loss:

```math
\mathcal{L}_{\mathrm{Charb}}(\hat{x}, x) =
\frac{\sqrt{\|\hat{x} - x\|_F^2 + \varepsilon^2} - \varepsilon}{\sqrt{\|x\|_F^2 + \varepsilon^2}}
+ \lambda_{\mathrm{lat}}\,\frac{\|\mathrm{encode}(nn, m, \hat{x}) - \mathrm{encode}(nn, m, x)\|_F^2}{\|x\|_F^2 + \varepsilon^2}.
```

Called as `loss(nn, m, x̂, x)`, it takes an external reconstruction ``\hat{x}``. Called as `loss(nn, m, x)` (or `loss(nn, x)` for the first modality), it uses ``\hat{x} = \mathrm{decode}(nn, m, \mathrm{encode}(nn, m, x))``.

Both forms compare ``\hat{x}`` with ``x`` in the form that `decode` returns: a pre-split pair `(q = F⁺, p = F⁻)` of a [`BilinearStiefelLayer`](@ref) modality, passed as ``x`` or as ``\hat{x}``, is recombined into the full field. The two halves of such a pair have one element type; halves of different element types raise an `ArgumentError`.

Here ``\varepsilon > 0`` (default `1e-3`) regularizes the Frobenius norm at ``\hat{x} = x`` and ``x = 0``, and ``\lambda_{\mathrm{lat}} \ge 0`` (default `1e-2`) weights the latent re-encoding consistency penalty.

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
    _charbonnier_eval(loss, _reconstruction_target(model, modality, x̂),
        _reconstruction_target(model, modality, x), z_hat, z)::T
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
    _charbonnier_eval(loss, x̂, _reconstruction_target(model, modality, x), z_hat, z)::T
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
    IsometricRolloutLoss(integrator; α_lat = 1.0, α_rec = 1.0, α_H = 0.0, H = nothing)

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

where ``z_k = \mathrm{encode}(nn, m, x_k)``, ``\hat{z}_1 = z_1``, ``\hat{z}_{k+1} = \mathrm{integrator}(\hat{z}_k)`` for ``k = 1, \dots, K - 1``, and `H` is a callable latent Hamiltonian.
`H(ẑ)` takes the latent state as `encode` and `integrator` give it, a vector of length ``2n`` or a matrix of size ``(2n, B)`` for a batched snapshot, and returns one scalar, one value for a whole batch.
The decoded state is compared with ``x_k`` in the form that `decode` returns: a pre-split pair `(q = F⁺, p = F⁻)` of a [`BilinearStiefelLayer`](@ref) modality is recombined into the full field.
The weights default to `α_lat = 1.0`, `α_rec = 1.0`, and `α_H = 0.0`, so `IsometricRolloutLoss(integrator)` has no energy term.

`H` has no default: the loss requires a latent Hamiltonian `H` when `α_H > 0`, and the constructor raises an `ArgumentError` without one. A fixed latent energy such as ``\frac{1}{2}\|\hat{z}\|_2^2`` is not the energy of the system. With `α_H = 0`, the loss has no energy term and takes no `H`: a given `H` would have no effect, so the constructor raises an `ArgumentError`.
The data, the output of `integrator` and, with `α_H > 0`, the value of `H` have the element type of the network's parameters; another element type raises an `ArgumentError`.
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
            α_H::Real = 0.0,
            H = nothing)
        (α_lat >= 0 && α_rec >= 0 && α_H >= 0) || throw(ArgumentError(
            "IsometricRolloutLoss requires non-negative weights α_lat, α_rec, α_H, got α_lat = $α_lat, α_rec = $α_rec, α_H = $α_H."))
        (iszero(α_H) || H !== nothing) || throw(ArgumentError(
            "IsometricRolloutLoss requires a latent Hamiltonian H when α_H > 0, got α_H = $α_H and H = nothing. Pass H, or set α_H = 0."))
        (!iszero(α_H) || H === nothing) || throw(ArgumentError(
            "IsometricRolloutLoss: H has no effect when α_H = 0, got α_H = $α_H and a latent Hamiltonian H. Set α_H > 0, or pass no H."))
        T = promote_type(typeof(float(α_lat)), typeof(float(α_rec)), typeof(float(α_H)))
        new{typeof(integrator), typeof(H), T}(
            integrator, H, T(α_lat), T(α_rec), T(α_H))
    end
end

# The value of `H` at the first latent state, and the penalty `|H(ẑ) - H(ẑ₁)|²` of a later one.
# `H = nothing` occurs only with `α_H = 0`, and then `H` is never called.
_initial_latent_H(::Nothing, z, ::Type{T}) where {T} = nothing
function _initial_latent_H(H, z, ::Type{T}) where {T}
    H1 = H(z)
    _check_eltype("IsometricRolloutLoss", "the value of H", H1, T)
    H1
end

_latent_H_penalty(::Nothing, z, ::Nothing, ::Type{T}) where {T} = zero(T)
function _latent_H_penalty(H, z, H1, ::Type{T}) where {T}
    Hk = H(z)
    _check_eltype("IsometricRolloutLoss", "the value of H", Hk, T)
    abs2(Hk - H1)
end

function _check_traj_len(K::Int)
    K >= 2 || throw(ArgumentError(
        "IsometricRolloutLoss requires at least 2 trajectory steps, got $K."))
    nothing
end

ChainRulesCore.@non_differentiable _check_traj_len(::Any)

_component_eltypes(x::Number) = (typeof(x),)
_component_eltypes(x::AbstractArray) = (eltype(x),)
_component_eltypes(x::NamedTuple{(:q, :p)}) = (eltype(x.q), eltype(x.p))
_component_eltypes(x::NamedTuple{(:X, :S, :V)}) = (eltype(x.X), eltype(x.S), eltype(x.V))

function _check_eltype(loss_name::String, what::String, x, ::Type{T}) where {T}
    for S in _component_eltypes(x)
        S === T || throw(ArgumentError(
            "$loss_name: $what has element type $S, but the network has element type $T."))
    end
    nothing
end

ChainRulesCore.@non_differentiable _check_eltype(::Any, ::Any, ::Any, ::Any)

Base.@constprop :aggressive function (loss::IsometricRolloutLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple,
        modality::Symbol, xs::AbstractVector)
    K = length(xs)
    _check_traj_len(K)
    T = _leaf_eltype(ps.core)
    _check_eltype("IsometricRolloutLoss", "the data", xs[1], T)
    z1 = encode(model, ps, modality, xs[1])
    H1 = _initial_latent_H(loss.H, z1, T)
    z_curr = z1
    err_lat = zero(T)
    err_rec = zero(T)
    err_H = zero(T)
    for k in 2:K
        _check_eltype("IsometricRolloutLoss", "the data", xs[k], T)
        z_k = encode(model, ps, modality, xs[k])
        z_curr = loss.integrator(z_curr)
        _check_eltype("IsometricRolloutLoss", "the integrator's output", z_curr, T)
        err_lat = err_lat + _frob_sq_diff(z_curr, z_k)
        x_rec_k = decode(model, ps, modality, z_curr)
        err_rec = err_rec +
                  _frob_sq_diff(x_rec_k, _reconstruction_target(model, modality, xs[k]))
        err_H = err_H + _latent_H_penalty(loss.H, z_curr, H1, T)
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
    CrossModalLoss(mod_a::Symbol, mod_b::Symbol; β_align = 1.0, β_cross = 1.0)

Cross-modal paired latent alignment and cross-reconstruction loss between modalities `mod_a` and `mod_b`:

```math
\mathcal{L}_{\mathrm{cross}}(x_a, x_b) =
\beta_{\mathrm{align}}\,\|z_a - z_b\|_F^2
+ \beta_{\mathrm{cross}}\,\|\mathrm{decode}(nn, b, z_a) - x_b\|_F^2,
```

where ``z_a = \mathrm{encode}(nn, a, x_a)`` and ``z_b = \mathrm{encode}(nn, b, x_b)``.
The pair ``(x_a, x_b)`` is the same physical state in two modalities: the alignment term asks that both encode to one latent state, and the cross-reconstruction term asks that the latent state of ``x_a`` decodes to ``x_b``.
The loss is not symmetric in ``a`` and ``b``: the cross-reconstruction term decodes only into modality `mod_b`, so `CrossModalLoss(mod_b, mod_a)` is another loss.
`decode(nn, b, z_a)` is compared with ``x_b`` in the form that `decode` returns: a pre-split pair `(q = F⁺, p = F⁻)` of a [`BilinearStiefelLayer`](@ref) modality is recombined into the full field.
A batch is the sum of its samples' losses; `x_a` and `x_b` with different numbers of samples raise a `DimensionMismatch`.

The weights are non-negative and default to `β_align = 1.0` and `β_cross = 1.0`; a negative or `NaN` weight raises an `ArgumentError`.
`x_a` and `x_b` have the element type of the network's parameters; another element type raises an `ArgumentError`.
"""
struct CrossModalLoss{T <: Real} <: NetworkLoss
    mod_a::Symbol
    mod_b::Symbol
    β_align::T
    β_cross::T

    function CrossModalLoss(
            mod_a::Symbol, mod_b::Symbol;
            β_align::Real = 1.0, β_cross::Real = 1.0)
        (β_align >= 0 && β_cross >= 0) || throw(ArgumentError(
            "CrossModalLoss requires non-negative weights β_align and β_cross, got β_align = $β_align, β_cross = $β_cross."))
        T = promote_type(typeof(float(β_align)), typeof(float(β_cross)))
        new{T}(mod_a, mod_b, T(β_align), T(β_cross))
    end
end

function _check_sample_count(z_a, z_b)
    size(z_a) == size(z_b) || throw(DimensionMismatch(
        "CrossModalLoss: the latent states of x_a and x_b have different sizes, $(size(z_a)) and $(size(z_b))."))
    nothing
end

ChainRulesCore.@non_differentiable _check_sample_count(::Any, ::Any)

Base.@constprop :aggressive function _cross_modal_eval(
        loss::CrossModalLoss, model::MultiModalSymplecticLayer,
        ps::NamedTuple, x_a, x_b)
    T = _leaf_eltype(ps.core)
    _check_eltype("CrossModalLoss", "x_a", x_a, T)
    _check_eltype("CrossModalLoss", "x_b", x_b, T)
    z_a = encode(model, ps, loss.mod_a, x_a)
    z_b = encode(model, ps, loss.mod_b, x_b)
    _check_sample_count(z_a, z_b)
    x̂_b = decode(model, ps, loss.mod_b, z_a)
    target = _reconstruction_target(model, loss.mod_b, x_b)
    res = T(loss.β_align) * _frob_sq_diff(z_a, z_b) +
          T(loss.β_cross) * _frob_sq_diff(x̂_b, target)
    res::T
end

Base.@constprop :aggressive function (loss::CrossModalLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple,
        x_a::ArrayOrNamedTuple, x_b::ArrayOrNamedTuple)
    _cross_modal_eval(loss, model, ps, x_a, x_b)
end

Base.@constprop :aggressive function (loss::CrossModalLoss)(
        model::MultiModalSymplecticLayer, ps::NetworkParameters,
        x_a::ArrayOrNamedTuple, x_b::ArrayOrNamedTuple)
    _cross_modal_eval(loss, model, params(ps), x_a, x_b)
end

Base.@constprop :aggressive function (loss::CrossModalLoss)(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder},
        x_a::ArrayOrNamedTuple, x_b::ArrayOrNamedTuple)
    loss(nn.model, params(nn), x_a, x_b)
end

Base.@constprop :aggressive function (loss::CrossModalLoss)(
        model::MultiModalSymplecticLayer, ps::NamedTuple, pair::Tuple{Any, Any})
    _cross_modal_eval(loss, model, ps, pair[1], pair[2])
end

Base.@constprop :aggressive function (loss::CrossModalLoss)(
        model::MultiModalSymplecticLayer, ps::NetworkParameters, pair::Tuple{Any, Any})
    _cross_modal_eval(loss, model, params(ps), pair[1], pair[2])
end

Base.@constprop :aggressive function (loss::CrossModalLoss)(
        nn::NeuralNetwork{<:MultiModalSymplecticAutoencoder}, pair::Tuple{Any, Any})
    loss(nn.model, params(nn), pair)
end
