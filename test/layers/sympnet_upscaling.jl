using GeometricMachineLearning, Test, Zygote
# `layer(chain, i)` is AbstractNeuralNetworks' accessor. This package adds no method to it and does
# not re-export it.
using AbstractNeuralNetworks: layer
import Random

Random.seed!(1234)

# This computes the Jacobians of the encoder, the shear pair and the decoder, and asserts the
# three exact symplectic identities the upscaling chain is built from -- not the round-trip
# composition, which is only *approximately* symplectic and is neither computed nor asserted here
# (the embedded Poisson tensor `E𝕁_NE'` has rank `N < N2`, so it cannot equal the full-rank
# `𝕁_{N2}`; see the `sympnet_upscaling.jl` entry under `[0.8.0]`, `### Infrastructure`, in
# `CHANGELOG.md` for the measured round-trip error).
function test_symplecticity(N = 4, N2 = 20, T = Float32)
    model = Chain(PSDLayer(N, N2), GradientLayerQ(N2, 2*N2, tanh),
        GradientLayerP(N2, 2*N2, tanh), PSDLayer(N2, N))
    ps = NeuralNetwork(model, CPU(), T).params
    x = rand(T, N)

    𝕁_N = PoissonTensor(N, T)
    𝕁_N2 = PoissonTensor(N2, T)

    # Encoder identity: PSDLayer is linear, so its Jacobian is the map itself.
    E = Zygote.jacobian(x -> layer(model, 1)(x, ps[1]), x)[1]
    @test isapprox(E' * 𝕁_N2 * E, 𝕁_N, atol = 1e-5)

    # Shear identity: the two GradientLayers are exactly symplectic at any point.
    G = Zygote.jacobian(
        y -> layer(model, 3)(layer(model, 2)(y, ps[2]), ps[3]),
        rand(T, N2))[1]
    @test isapprox(G' * 𝕁_N2 * G, 𝕁_N2, atol = 1e-5)

    # Decoder identity: PSDLayer is symmetric in construction, so the same identity holds in the
    # other direction, with the transpose on the other side to match the N2 -> N shape.
    Dec = Zygote.jacobian(y -> layer(model, 4)(y, ps[4]), rand(T, N2))[1]
    @test isapprox(Dec * 𝕁_N2 * Dec', 𝕁_N, atol = 1e-5)
end

# `PSDLayer{M, N}` and `GradientLayer{M, N, …}` carry the sizes in their types, so each new pair
# `(N, N2)` is a fresh compilation of the whole chain. These 12 pairs, out of the 120 with `N` in
# `2:2:20` and `N2` in `(2N):2:(4N)`, cover the edges: the smallest pair `(2, 4)`, `N2 = 2N`,
# `N2 = 4N`, `N2` not a multiple of `N`, and the largest `N = 20`.
for (N, N2) in ((2, 4), (2, 6), (2, 8), (4, 10), (6, 12), (6, 20),
    (8, 24), (10, 30), (12, 36), (14, 42), (20, 40), (20, 80))
    test_symplecticity(N, N2)
end
