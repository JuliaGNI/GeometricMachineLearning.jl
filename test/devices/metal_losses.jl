# `ReducedHamiltonianLoss` and `LatentVectorFieldLoss` on an Apple GPU, under `allowscalar(false)`.
# Their central differences shift the latent points by a shift on the device of the data, and the energy
# term copies a reference state of the host onto that device. The value on the device is the value on
# the host, and the parameter gradient is on the device. `test/devices/jlarray_losses.jl` checks the
# same on a `JLArray`; this file runs the kernels that Metal compiles. `Float32` only, because Metal has
# no `Float64`. Where `Metal.functional()` is false, the file records one skip.

using GeometricMachineLearning
using GeometricMachineLearning: params
using AbstractNeuralNetworks: changebackend
using GPUArraysCore: allowscalar
using KernelAbstractions: get_backend
using Metal: Metal, MtlArray
using Test
import Random
import Zygote

const T = Float32

# |p|²/2 + cos(q₁) + q₂²/2 on ℝ⁴, and its vector field 𝕁∇H.
H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ cos.(X[1, :]) .+ X[2, :] .^ 2 ./ 2
XH(X) = vcat(X[3:4, :], sin.(X[1:1, :]), -X[2:2, :])

const ARCH = SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)

# The device sums its products in another order than the host. The central differences with the
# default step cbrt(eps(T)) divide that round-off by the step, an error of order eps(T)^(2/3).
const RTOL = 20 * cbrt(eps(T))^2

# The number of array leaves of the gradient `g`, and whether each is a device array of element type
# `T`. A leaf may itself be a `NamedTuple` of arrays, and a leaf a network does not train is `nothing`.
function device_leaves(g, ::Type{AT}) where {AT}
    n, ok = Ref(0), Ref(true)
    walk(x::NamedTuple) = foreach(walk, values(x))
    walk(x::AbstractArray) = (n[] += 1; ok[] &= parent(x) isa AT)
    walk(::Nothing) = nothing
    for name in keys(g), k in keys(g[name])

        walk(g[name][k])
    end
    n[], ok[]
end

if Metal.functional()
    allowscalar(false)
    Random.seed!(123)
    nn = NeuralNetwork(ARCH, CPU(), T)
    model, ps = nn.model, params(nn)
    ps_device = changebackend(get_backend(MtlArray(zeros(T, 1))), ps)
    x = randn(T, 4, 6)
    losses = (
        "LatentVectorFieldLoss" => (
            LatentVectorFieldLoss(ARCH, H; λ = 0.7, energy = true,
                reference_state = [0.3, -0.2, 0.1, 0.4]),
            x, XH(x)),
        "ReducedHamiltonianLoss" => (ReducedHamiltonianLoss(H; λ = 0.7), x, vcat(x, XH(x))))
    @testset "$name on Metal" for (name, (loss, input, output)) in losses
        l_host = loss(model, ps, input, output)
        l_device = loss(model, ps_device, MtlArray(input), MtlArray(output))
        @test l_device isa T
        @test l_device ≈ l_host rtol = RTOL
        g = Zygote.gradient(p -> loss(model, p, MtlArray(input), MtlArray(output)), ps_device)[1]
        n_leaves, on_device = device_leaves(g, MtlArray{T})
        @test n_leaves > 0
        @test on_device
    end
else
    @test_skip Metal.functional()
end
