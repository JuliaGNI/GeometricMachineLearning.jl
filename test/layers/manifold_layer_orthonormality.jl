using Test, GeometricMachineLearning
import Random

Random.seed!(1234)

# `StiefelLayer`, `GrassmannLayer` and `PSDLayer` orthonormalize the weight they initialise. They
# do it through `GeometricOptimizers.orthonormal_columns`, which is CholeskyQR2 and runs on
# whatever backend the draw was allocated on; `LinearAlgebra.qr!` is a host factorization and
# throws on a device. `check(Y)` is ‖YᵀY − I‖, so what is below is the invariant either
# factorization has to keep. A device is what distinguishes the two and no test here runs on one --
# this catches a host-side regression in the initialisation.
#
# The bound is `100 * eps(T)`. Two independent sweeps over the cases below put the worst residual
# at `3.7 * eps(T)` and at `4.3 * eps(T)`, each over 300 draws per element type, so the figure is
# a sample and not a ceiling. The margin is wide for that reason, and because CholeskyQR2 redraws
# when the Gram matrix breaks Cholesky down, so the factor is not a fixed function of the draw.

# Each layer's `initialparameters` allocates `N > M ? (N, M) : (M, N)`, so swapping the two
# arguments reaches both branches. `PSDLayer` halves both dimensions, which is why they are even.
const LAYERS = ((StiefelLayer, 4, 10), (StiefelLayer, 10, 4),
    (GrassmannLayer, 4, 10), (GrassmannLayer, 10, 4),
    (PSDLayer, 4, 10), (PSDLayer, 10, 4))

for T in (Float32, Float64)
    for (layer, first_dim, second_dim) in LAYERS
        weight = NeuralNetwork(Chain(layer(first_dim, second_dim)), T).params.L1.weight
        @test eltype(weight) == T
        @test check(weight) < 100 * eps(T)
    end
end
