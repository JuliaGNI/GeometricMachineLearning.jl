# The forward pass of `WideResNetLayer` and of `ParametricResNetLayer`, written out with `*` and
# `reshape`. This is the oracle of the tests of `src/layers/wide_resnet.jl`,
# `src/layers/parametric_resnet_layer.jl` and `src/architectures/parametric_resnet.jl`: it goes
# through neither `_mul` nor `concatenate_array_with_parameters`, so a layer that drops the residual
# input or the system parameters cannot reproduce it.

# `x ↦ x + σ.(W_down σ.(W_up x .+ b_up) .+ b)`, for every sample of `x` at once.
function wide_resnet_forward(x::AbstractArray, ps, σ)
    flat = reshape(x, size(x, 1), :)
    y = flat + σ.(ps.downscale_weight * σ.(ps.upscale_weight * flat .+ ps.upscale_bias) .+
           ps.bias)
    reshape(y, size(x))
end

# `x ↦ x + σ.(W_down σ.(W_up [x; μ] .+ b_up) .+ b)`, with the flattened `μ` below every sample.
function parametric_forward(x::AbstractArray, μ::AbstractVector, ps, σ)
    flat = reshape(x, size(x, 1), :)
    input = vcat(flat, repeat(reshape(μ, :, 1), 1, size(flat, 2)))
    y = flat + σ.(ps.downscale_weight * σ.(ps.upscale_weight * input .+ ps.upscale_bias) .+
           ps.bias)
    reshape(y, size(x))
end
