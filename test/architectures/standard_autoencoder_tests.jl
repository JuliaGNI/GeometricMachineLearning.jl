using GeometricMachineLearning
using GeometricMachineLearning: params
using Test
using Zygote: jacobian
import Random

Random.seed!(123)

function test_encoder_and_decoder(N::Integer, n::Integer; kwargs...)
    nn = NeuralNetwork(StandardAutoencoder(N, n; kwargs...))
    ae_encoder = encoder(nn)
    ae_decoder = decoder(nn)

    test_vec = rand(N)
    test_mat = rand(N, N)

    @test size(ae_encoder(test_vec)) == (n,)
    @test size(ae_decoder(rand(n))) == (N,)
    @test nn(test_vec) ≈ ae_decoder(ae_encoder(test_vec))
    @test nn(test_mat) ≈ ae_decoder(ae_encoder(test_mat))
end

# Each nonlinear layer is one Dense layer with bias, and each part ends in an affine layer.
function test_parameter_count(N::Integer, n::Integer, w::Integer, n_enc::Integer, n_dec::Integer)
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

function test_training_reduces_loss(N::Integer, n::Integer; n_epochs::Integer = 100)
    dl = DataLoader(rand(N, 10 * N); autoencoder = true)
    nn = NeuralNetwork(StandardAutoencoder(N, n))
    loss = AutoEncoderLoss()
    loss_before = loss(nn, dl.input)
    o = Optimizer(Adam(), nn)
    o(nn, dl, Batch(10), n_epochs; show_progress = false)
    @test loss(nn, dl.input) < loss_before
end

# Nothing makes the decoder symplectic: ∇Ψᵀ 𝕁 ∇Ψ differs from 𝕁 at a random point.
function test_not_symplectic(N::Integer, n::Integer)
    ae_decoder = decoder(NeuralNetwork(StandardAutoencoder(N, n)))
    J = jacobian(ae_decoder, rand(n))[1]
    @test !(J' * PoissonTensor(N) * J ≈ PoissonTensor(n))
end

for (N, n) in ((10, 4), (4, 2), (7, 3))
    test_encoder_and_decoder(N, n)
    test_parameter_count(N, n, 2N, 2, 2)
end
test_encoder_and_decoder(4, 2; width = 16, n_encoder_layers = 1, n_decoder_layers = 4)
test_parameter_count(4, 2, 40, 3, 4)
test_training_reduces_loss(10, 4)
test_not_symplectic(10, 4)
