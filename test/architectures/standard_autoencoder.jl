using GeometricMachineLearning
using Test
using Zygote: jacobian
import Random

include("../helpers/eltypes.jl")
include("../helpers/poisson.jl")

Random.seed!(123)

function test_encoder_and_decoder(::Type{T}, N::Integer, n::Integer; kwargs...) where {T}
    nn = NeuralNetwork(StandardAutoencoder(N, n; kwargs...), T)
    ae_encoder = encoder(nn)
    ae_decoder = decoder(nn)

    test_vec = rand(T, N)
    test_mat = rand(T, N, N)

    @test size(ae_encoder(test_vec)) == (n,)
    @test size(ae_decoder(rand(T, n))) == (N,)
    @test eltype(nn(test_mat)) === T
    @test nn(test_vec) ≈ ae_decoder(ae_encoder(test_vec))
    @test nn(test_mat) ≈ ae_decoder(ae_encoder(test_mat))
end

# Each nonlinear layer is one Dense layer with bias, and each part ends in an affine layer.
function test_parameter_count(
        N::Integer, n::Integer, w::Integer, n_enc::Integer, n_dec::Integer)
    nn = NeuralNetwork(StandardAutoencoder(N, n; width = w,
        n_encoder_layers = n_enc, n_decoder_layers = n_dec))
    n_encoder = (N + 1) * w + (n_enc - 1) * (w + 1) * w + (w + 1) * n
    n_decoder = (n + 1) * w + (n_dec - 1) * (w + 1) * w + (w + 1) * N
    @test parameterlength(nn.model) == n_encoder + n_decoder
    @test parameterlength(encoder(nn).model) == n_encoder
    @test parameterlength(decoder(nn).model) == n_decoder
    @test length(encoder(nn).model.layers) == n_enc + 1
    @test length(decoder(nn).model.layers) == n_dec + 1
end

# One `train!` step moves every parameter and gives a finite loss of the network's element type.
function test_one_training_step(::Type{T}, N::Integer, n::Integer) where {T}
    n_samples = 2N
    dl = DataLoader(rand(T, N, n_samples); autoencoder = true)
    nn = NeuralNetwork(StandardAutoencoder(N, n), T)
    before = deepcopy(nn.params)
    o = TrainingOptimizer(nn.params; algorithm = Adam())
    history = train!(nn, o, dl, Batch(n_samples), 1; show_progress = false)
    @test eltype(history) === T
    @test all(isfinite, history)
    for layer in keys(nn.params), leaf in (:W, :b)

        @test getproperty(nn.params, layer)[leaf] != getproperty(before, layer)[leaf]
    end
end

# Nothing makes the decoder symplectic: ∇Ψᵀ 𝕁 ∇Ψ differs from 𝕁 at a random point.
function test_not_symplectic(::Type{T}, N::Integer, n::Integer) where {T}
    ae_decoder = decoder(NeuralNetwork(StandardAutoencoder(N, n), T))
    J = jacobian(ae_decoder, rand(T, n))[1]
    @test !(J' * poisson_matrix(N, T) * J ≈ poisson_matrix(n, T))
end

@testset "StandardAutoencoder rejects invalid dimensions and keywords" begin
    re_reduced = r"^ArgumentError: reduced_dim \(0\) must be at least 1\.$"
    @test_throws re_reduced StandardAutoencoder(4, 0)
    @test_throws re_reduced StandardAutoencoder(0, 0)
    re_full = r"^ArgumentError: full_dim \(4\) must be greater than or equal to reduced_dim \(6\)\.$"
    @test_throws re_full StandardAutoencoder(4, 6)
    @test_throws r"^ArgumentError: width \(0\) must be at least 1\.$" StandardAutoencoder(
        4, 2; width = 0)
    @test_throws r"^ArgumentError: width \(-3\) must be at least 1\.$" StandardAutoencoder(
        4, 2; width = -3)
    re_layers = r"^ArgumentError: n_encoder_layers \(0\) and n_decoder_layers \(2\) must be at least 1\.$"
    @test_throws re_layers StandardAutoencoder(4, 2; n_encoder_layers = 0)
    re_layers_dec = r"^ArgumentError: n_encoder_layers \(2\) and n_decoder_layers \(0\) must be at least 1\.$"
    @test_throws re_layers_dec StandardAutoencoder(4, 2; n_decoder_layers = 0)
end

# The defaults are `width = 2full_dim`, two nonlinear `tanh` layers in each part, and an identity
# output layer; a non-default activation reaches every nonlinear layer and not the output layer.
@testset "StandardAutoencoder builds its layers from the keywords" begin
    N, n = 7, 3
    arch = StandardAutoencoder(N, n)
    @test (arch.width, arch.n_encoder_layers, arch.n_decoder_layers) == (2N, 2, 2)
    nn = NeuralNetwork(arch)
    @test encoder(nn).model.layers ==
          (Dense(N, 2N, tanh), Dense(2N, 2N, tanh), Dense(2N, n, identity))
    @test decoder(nn).model.layers ==
          (Dense(n, 2N, tanh), Dense(2N, 2N, tanh), Dense(2N, N, identity))

    nn_sin = NeuralNetwork(StandardAutoencoder(N, n; width = 5, n_encoder_layers = 3,
        n_decoder_layers = 1, activation = sin))
    @test encoder(nn_sin).model.layers ==
          (Dense(N, 5, sin), Dense(5, 5, sin), Dense(5, 5, sin), Dense(5, n, identity))
    @test decoder(nn_sin).model.layers == (Dense(n, 5, sin), Dense(5, N, identity))
end

@testset "StandardAutoencoder ($T)" for T in REAL_ELTYPES
    for (N, n) in ((10, 4), (4, 2), (7, 3), (4, 4), (1, 1))
        test_encoder_and_decoder(T, N, n)
        test_parameter_count(N, n, 2N, 2, 2)
    end
    test_encoder_and_decoder(
        T, 4, 2; width = 16, n_encoder_layers = 1, n_decoder_layers = 4)
    test_parameter_count(4, 2, 40, 3, 4)
    test_one_training_step(T, 10, 4)
    test_not_symplectic(T, 10, 4)
end
