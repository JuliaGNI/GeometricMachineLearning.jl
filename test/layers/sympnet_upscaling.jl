using GeometricMachineLearning, Test, Zygote

# This computes the Jacobians of the encoder, the shear pair and the decoder, and asserts the
# three exact symplectic identities the upscaling chain is built from -- not the round-trip
# composition, which is only *approximately* symplectic and is neither computed nor asserted here
# (the embedded Poisson tensor `E𝕁_NE'` has rank `N < N2`, so it cannot equal the full-rank
# `𝕁_{N2}`; see the `sympnet_upscaling.jl` entry under `[Unreleased]`, `### Infrastructure`, in
# `CHANGELOG.md` for the measured round-trip error).
function test_symplecticity(N = 4, N2 = 20, T = Float32)
    model = Chain(PSDLayer(N, N2), GradientLayerQ(N2, 2*N2, tanh),
        GradientLayerP(N2, 2*N2, tanh), PSDLayer(N2, N))
    ps = NeuralNetwork(model, CPU(), T).params
    x = rand(T, N)

    𝕁_N = PoissonTensor(N, T)
    𝕁_N2 = PoissonTensor(N2, T)

    # Encoder identity: PSDLayer is linear, so its Jacobian is the map itself.
    E = Zygote.jacobian(x -> GeometricMachineLearning.layer(model, 1)(x, ps[1]), x)[1]
    @test isapprox(E' * 𝕁_N2 * E, 𝕁_N, atol = 1e-5)

    # Shear identity: the two GradientLayers are exactly symplectic at any point.
    G = Zygote.jacobian(
        y -> GeometricMachineLearning.layer(model, 3)(
            GeometricMachineLearning.layer(model, 2)(y, ps[2]), ps[3]),
        rand(T, N2))[1]
    @test isapprox(G' * 𝕁_N2 * G, 𝕁_N2, atol = 1e-5)

    # Decoder identity: PSDLayer is symmetric in construction, so the same identity holds in the
    # other direction, with the transpose on the other side to match the N2 -> N shape.
    Dec = Zygote.jacobian(y -> GeometricMachineLearning.layer(model, 4)(y, ps[4]), rand(T, N2))[1]
    @test isapprox(Dec * 𝕁_N2 * Dec', 𝕁_N, atol = 1e-5)
end

for N in 2:2:20
    for N2 in (2 * N):2:(4 * N)
        test_symplecticity(N, N2)
    end
end
