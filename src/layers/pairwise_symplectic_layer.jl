@doc raw"""
    PairwiseSymplecticLayer{M, U, TA, C, Sign} <: SympNetLayer{M, M}

Permutation-equivariant pairwise symplectic shear layer on ``T^*(\mathbb{R}^d)^{M_p} \cong \mathbb{R}^{M}`` with total phase-space dimension `M = 2 * M_p * d` and upscaled per-particle feature dimension `U = upscale * d`.

See [`PairwiseSymplecticLayerQ`](@ref), [`PairwiseSymplecticLayerP`](@ref), [`InversePairwiseSymplecticLayerQ`](@ref), and [`InversePairwiseSymplecticLayerP`](@ref).
"""
struct PairwiseSymplecticLayer{M, U, TA, C, Sign} <: SympNetLayer{M, M}
    n_particles::Int
    particle_dim::Int
    upscale_dim::Int
    activation::TA
end

@doc raw"""
    PairwiseSymplecticLayerQ(M, d, upscale, activation = tanh)

Make an instance of an ``S_M``-equivariant pairwise symplectic layer that updates canonical momenta ``p`` via self and pairwise relative potential forces:

```math
p_i^{\mathrm{new}} = p_i - K_{\mathrm{self}}^T \sigma(K_{\mathrm{self}} q_i + b_{\mathrm{self}})
- \frac{1}{M}\sum_{j=1}^M K_{\mathrm{pair}}^T \tau\bigl(K_{\mathrm{pair}}(q_i - q_j)\bigr),
```
where ``\tau(x) = \sigma(x) - \sigma(-x)`` is oddized so that the pairwise force ``f_{ij} = -f_{ji}`` is skew-symmetric in ``(i, j)`` and conserves total canonical momentum ``\sum_{i=1}^M p_i`` when ``K_{\mathrm{self}} = 0``.
"""
const PairwiseSymplecticLayerQ{M, U, TA} = PairwiseSymplecticLayer{M, U, TA, :Q, -1}

@doc raw"""
    PairwiseSymplecticLayerP(M, d, upscale, activation = tanh)

Make an instance of an ``S_M``-equivariant pairwise symplectic layer that updates canonical positions ``q`` via self and pairwise relative velocity differences ``p_i - p_j``:

```math
q_i^{\mathrm{new}} = q_i + K_{\mathrm{self}}^T \sigma(K_{\mathrm{self}} p_i + b_{\mathrm{self}})
+ \frac{1}{M}\sum_{j=1}^M K_{\mathrm{pair}}^T \tau\bigl(K_{\mathrm{pair}}(p_i - p_j)\bigr),
```
where ``\tau(x) = \sigma(x) - \sigma(-x)``.
"""
const PairwiseSymplecticLayerP{M, U, TA} = PairwiseSymplecticLayer{M, U, TA, :P, +1}

@doc raw"""
    InversePairwiseSymplecticLayerQ(M, d, upscale, activation = tanh)

Analytic symplectic inverse of [`PairwiseSymplecticLayerQ`](@ref), flipping the sign of the momentum force update while sharing the identical parameter `NamedTuple` `ps`.
"""
const InversePairwiseSymplecticLayerQ{M, U, TA} = PairwiseSymplecticLayer{M, U, TA, :Q, +1}

@doc raw"""
    InversePairwiseSymplecticLayerP(M, d, upscale, activation = tanh)

Analytic symplectic inverse of [`PairwiseSymplecticLayerP`](@ref), flipping the sign of the position displacement update while sharing the identical parameter `NamedTuple` `ps`.
"""
const InversePairwiseSymplecticLayerP{M, U, TA} = PairwiseSymplecticLayer{M, U, TA, :P, -1}

function _check_pairwise_args(M::Integer, d::Integer, upscale::Integer)
    (M >= 1 && d >= 1 && upscale >= 1) ||
        throw(ArgumentError(
            "PairwiseSymplecticLayer requires M >= 1, d >= 1, and upscale >= 1, got M = $M, d = $d, upscale = $upscale."))
    nothing
end

function PairwiseSymplecticLayerQ(
        M::Integer, d::Integer, upscale::Integer, activation = tanh)
    _check_pairwise_args(M, d, upscale)
    Mp, dp, up = Int(M), Int(d), Int(upscale)
    U = up * dp
    PairwiseSymplecticLayer{2 * Mp * dp, U, typeof(activation), :Q, -1}(
        Mp, dp, U, activation)
end

function PairwiseSymplecticLayerP(
        M::Integer, d::Integer, upscale::Integer, activation = tanh)
    _check_pairwise_args(M, d, upscale)
    Mp, dp, up = Int(M), Int(d), Int(upscale)
    U = up * dp
    PairwiseSymplecticLayer{2 * Mp * dp, U, typeof(activation), :P, +1}(
        Mp, dp, U, activation)
end

function InversePairwiseSymplecticLayerQ(
        M::Integer, d::Integer, upscale::Integer, activation = tanh)
    _check_pairwise_args(M, d, upscale)
    Mp, dp, up = Int(M), Int(d), Int(upscale)
    U = up * dp
    PairwiseSymplecticLayer{2 * Mp * dp, U, typeof(activation), :Q, +1}(
        Mp, dp, U, activation)
end

function InversePairwiseSymplecticLayerP(
        M::Integer, d::Integer, upscale::Integer, activation = tanh)
    _check_pairwise_args(M, d, upscale)
    Mp, dp, up = Int(M), Int(d), Int(upscale)
    U = up * dp
    PairwiseSymplecticLayer{2 * Mp * dp, U, typeof(activation), :P, -1}(
        Mp, dp, U, activation)
end

function parameterlength(l::PairwiseSymplecticLayer)
    U = l.upscale_dim
    d = l.particle_dim
    2 * U * d + U
end

function initialparameters(
        rng::AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer,
        l::PairwiseSymplecticLayer, backend::KernelAbstractions.Backend, ::Type{T};
        init_bias = ZeroInitializer()) where {T}
    U = l.upscale_dim
    d = l.particle_dim
    K_self = KernelAbstractions.allocate(backend, T, U, d)
    b_self = KernelAbstractions.allocate(backend, T, U)
    K_pair = KernelAbstractions.allocate(backend, T, U, d)
    init_weight(rng, K_self)
    init_bias(rng, b_self)
    init_weight(rng, K_pair)
    (K_self = K_self, b_self = b_self, K_pair = K_pair)
end

function _pairwise_force_2d(
        l::PairwiseSymplecticLayer, u::AbstractMatrix{T}, ps::NamedTuple) where {T}
    Mp = l.n_particles
    inv_M = inv(T(Mp))
    σ = l.activation
    K_self = ps.K_self
    b_self = ps.b_self
    K_pair = ps.K_pair

    f_self = transpose(K_self) * σ.(K_self * u .+ b_self)
    Ku = K_pair * u
    diff = reshape(Ku, :, Mp, 1) .- reshape(Ku, :, 1, Mp)
    τ = σ.(diff) .- σ.(-diff)
    τ_mean = dropdims(sum(τ; dims = 3); dims = 3) .* inv_M
    f_pair = transpose(K_pair) * τ_mean
    f_self .+ f_pair
end

function _pairwise_force_3d(
        l::PairwiseSymplecticLayer, u::AbstractArray{T, 3}, ps::NamedTuple) where {T}
    d, Mp, B = size(u)
    U = l.upscale_dim
    inv_M = inv(T(Mp))
    σ = l.activation
    K_self = ps.K_self
    b_self = ps.b_self
    K_pair = ps.K_pair

    u_flat = reshape(u, d, Mp * B)
    f_self = reshape(transpose(K_self) * σ.(K_self * u_flat .+ b_self), d, Mp, B)
    Ku = reshape(K_pair * u_flat, U, Mp, B)
    diff = reshape(Ku, U, Mp, 1, B) .- reshape(Ku, U, 1, Mp, B)
    τ = σ.(diff) .- σ.(-diff)
    τ_mean = reshape(dropdims(sum(τ; dims = 3); dims = 3) .* inv_M, U, Mp * B)
    f_pair = reshape(transpose(K_pair) * τ_mean, d, Mp, B)
    f_self .+ f_pair
end

function _apply_pairwise_shift(::Val{:Q}, ::Val{-1}, q, p, Δ)
    (q = q, p = p .- Δ)
end
function _apply_pairwise_shift(::Val{:Q}, ::Val{+1}, q, p, Δ)
    (q = q, p = p .+ Δ)
end
function _apply_pairwise_shift(::Val{:P}, ::Val{+1}, q, p, Δ)
    (q = q .+ Δ, p = p)
end
function _apply_pairwise_shift(::Val{:P}, ::Val{-1}, q, p, Δ)
    (q = q .- Δ, p = p)
end

function (l::PairwiseSymplecticLayer{M, U, TA, C, Sign})(
        qp::NamedTuple{(:q, :p)}, ps::NamedTuple) where {M, U, TA, C, Sign}
    q, p = qp.q, qp.p
    Mp = l.n_particles
    dp = l.particle_dim
    N = Mp * dp
    if ndims(q) == 2
        (size(q) == size(p) && size(q) == (dp, Mp)) ||
            throw(DimensionMismatch(
                "PairwiseSymplecticLayer expects q and p of matching size ($dp, $Mp), ($dp, $Mp, B), or ($N,), got $(size(q)) and $(size(p))."))
        u = C === :Q ? q : p
        Δ = _pairwise_force_2d(l, u, ps)
        return _apply_pairwise_shift(Val(C), Val(Sign), q, p, Δ)
    elseif ndims(q) == 3
        (size(q) == size(p) && size(q, 1) == dp && size(q, 2) == Mp) ||
            throw(DimensionMismatch(
                "PairwiseSymplecticLayer expects q and p of matching size ($dp, $Mp), ($dp, $Mp, B), or ($N,), got $(size(q)) and $(size(p))."))
        u = C === :Q ? q : p
        Δ = _pairwise_force_3d(l, u, ps)
        return _apply_pairwise_shift(Val(C), Val(Sign), q, p, Δ)
    elseif ndims(q) == 1
        (size(q) == size(p) && length(q) == N) ||
            throw(DimensionMismatch(
                "PairwiseSymplecticLayer expects q and p of matching size ($dp, $Mp), ($dp, $Mp, B), or ($N,), got $(size(q)) and $(size(p))."))
        q_mat = reshape(q, dp, Mp)
        p_mat = reshape(p, dp, Mp)
        out = l((q = q_mat, p = p_mat), ps)
        return (q = vec(out.q), p = vec(out.p))
    else
        throw(DimensionMismatch(
            "PairwiseSymplecticLayer expects q and p of matching size ($dp, $Mp), ($dp, $Mp, B), or ($N,), got $(size(q)) and $(size(p))."))
    end
end

function (l::PairwiseSymplecticLayer{M})(
        z::AbstractVector, ps::NamedTuple) where {M}
    length(z) == M ||
        throw(DimensionMismatch(
            "PairwiseSymplecticLayer expects first dimension $M, got $(length(z))."))
    Mp = l.n_particles
    dp = l.particle_dim
    N = Mp * dp
    q = reshape(@view(z[1:N]), dp, Mp)
    p = reshape(@view(z[(N + 1):M]), dp, Mp)
    out = l((q = q, p = p), ps)
    vcat(vec(out.q), vec(out.p))
end

function (l::PairwiseSymplecticLayer{M})(
        z::AbstractMatrix, ps::NamedTuple) where {M}
    N2, B = size(z)
    N2 == M ||
        throw(DimensionMismatch(
            "PairwiseSymplecticLayer expects first dimension $M, got $N2."))
    Mp = l.n_particles
    dp = l.particle_dim
    N = Mp * dp
    q = reshape(@view(z[1:N, :]), dp, Mp, B)
    p = reshape(@view(z[(N + 1):M, :]), dp, Mp, B)
    out = l((q = q, p = p), ps)
    vcat(reshape(out.q, N, B), reshape(out.p, N, B))
end

function inverse_layer(l::PairwiseSymplecticLayer{
        M, U, TA, C, Sign}) where {M, U, TA, C, Sign}
    PairwiseSymplecticLayer{M, U, TA, C, -Sign}(
        l.n_particles, l.particle_dim, l.upscale_dim, l.activation)
end

inverse_layer(::Type{PairwiseSymplecticLayerQ}) = InversePairwiseSymplecticLayerQ
inverse_layer(::Type{InversePairwiseSymplecticLayerQ}) = PairwiseSymplecticLayerQ
inverse_layer(::Type{PairwiseSymplecticLayerP}) = InversePairwiseSymplecticLayerP
inverse_layer(::Type{InversePairwiseSymplecticLayerP}) = PairwiseSymplecticLayerP

function inverse_layer(::Type{PairwiseSymplecticLayer{
        M, U, TA, C, Sign}}) where {M, U, TA, C, Sign}
    PairwiseSymplecticLayer{M, U, TA, C, -Sign}
end
