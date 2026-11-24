# `ReducedHamiltonianLoss` and `LatentVectorFieldLoss` on a device array. Their central differences
# shift the latent points, and the energy term encodes a reference state that the loss stores on the
# host. Both have to stay on the device of the data: a host array in a broadcast with a `JLArray`
# raises an error. The loss on the device is the loss on the host, and its gradient is on the device.

using GeometricMachineLearning
using GeometricMachineLearning: params
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Zygote: Zygote
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/device.jl")

Random.seed!(123)
allowscalar(false)

# |p|²/2 + cos(q₁) + q₂²/2 on ℝ⁴, and its vector field 𝕁∇H.
H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ cos.(X[1, :]) .+ X[2, :] .^ 2 ./ 2
XH(X) = vcat(X[3:4, :], sin.(X[1:1, :]), -X[2:2, :])

const ARCH = SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)

# The device sums its products in another order than the host. The central differences with the
# default step cbrt(eps(T)) divide that round-off by the step, an error of order eps(T)^(2/3).
fd_rtol(::Type{T}) where {T} = 20 * cbrt(eps(T))^2

@testset "Both Hamiltonian losses on a JLArray, T = $T" for T in REAL_ELTYPES
    nn = NeuralNetwork(ARCH, CPU(), T)
    model, ps = nn.model, params(nn)
    ps_device = to_device(ps)
    x = randn(T, 4, 6)
    losses = (
        (
            LatentVectorFieldLoss(ARCH, H; λ = 0.7, energy = true,
                reference_state = [0.3, -0.2, 0.1, 0.4]),
            x,
            XH(x)),
        (ReducedHamiltonianLoss(H; λ = 0.7), x, vcat(x, XH(x))))
    for (loss, input, output) in losses
        l_host = loss(model, ps, input, output)
        l_device = loss(model, ps_device, JLArray(input), JLArray(output))
        @test l_device isa T
        @test l_device ≈ l_host rtol = fd_rtol(T)
        g = Zygote.gradient(p -> loss(model, p, JLArray(input), JLArray(output)), ps_device)[1]
        @test leaves_on_device(g, T)
    end
end
