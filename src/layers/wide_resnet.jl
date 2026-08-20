@doc raw"""
    WideResNetLayer(dim, width, activation)

A [`ResNetLayer`](@ref) whose hidden layer is `width` wide, independent of `dim`.

`ResNetLayer` applies a single ``\mathrm{dim} \times \mathrm{dim}`` weight, so its capacity is tied
to the system dimension. This one upscales to `width`, applies the activation, and downscales again:

```math
    x \mapsto x + \sigma(W_\mathrm{down}\sigma(W_\mathrm{up}x + b_\mathrm{up}) + b),
```

with ``W_\mathrm{up}\in\mathbb{R}^{\mathrm{width}\times\mathrm{dim}}`` and
``W_\mathrm{down}\in\mathbb{R}^{\mathrm{dim}\times\mathrm{width}}``. [`ResNet`](@ref) uses it
whenever the `width` it is given differs from the system dimension.

Also see [`ParametricResNetLayer`](@ref), which additionally takes the parameters of the system.
"""
struct WideResNetLayer{M, N, F1} <: AbstractExplicitLayer{M, N}
    width::Int
    activation::F1
end

WideResNetLayer(dim::Integer, width::Integer, activation=identity) = WideResNetLayer{dim, dim, typeof(activation)}(width, activation)

function initialparameters(rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer, l::WideResNetLayer{M, M}, backend::KernelAbstractions.Backend, ::Type{T}; init_bias = ZeroInitializer()) where {M, T}
    upscale_weight = KernelAbstractions.allocate(backend, T, l.width, M)
    upscale_bias = KernelAbstractions.allocate(backend, T, l.width)
    downscale_weight = KernelAbstractions.allocate(backend, T, M, l.width)
    bias = KernelAbstractions.allocate(backend, T, M)
    init_weight(rng, upscale_weight)
    init_weight(rng, downscale_weight)
    init_bias(rng, upscale_bias)
    init_bias(rng, bias)
    (upscale_weight=upscale_weight, downscale_weight=downscale_weight, upscale_bias=upscale_bias, bias=bias)
end

parameterlength(l::WideResNetLayer{M, M}) where {M} = l.width * (M + 1) + M * (l.width + 1)

(d::WideResNetLayer{M, M})(x::AbstractVecOrMat, ps::NamedTuple) where {M} = x + d.activation.(ps.downscale_weight * d.activation.(ps.upscale_weight * x .+ ps.upscale_bias) .+ ps.bias)

(d::WideResNetLayer{M, M})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, T} = x + d.activation.(mat_tensor_mul(ps.downscale_weight, d.activation.(mat_tensor_mul(ps.upscale_weight, x) .+ ps.upscale_bias)) .+ ps.bias)

function (d::WideResNetLayer{M, M})(z::QPT, ps::NamedTuple) where {M}
    @assert iseven(M)
    @assert size(z.q, 1) * 2 == M
    N2 = M ÷ 2
    output = d(vcat(z.q, z.p), ps)
    assign_q_and_p(output, N2)
end