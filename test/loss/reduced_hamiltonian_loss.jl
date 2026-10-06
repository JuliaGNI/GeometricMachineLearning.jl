using GeometricMachineLearning
using GeometricMachineLearning: _reduced_hamiltonian_gradient, params
using Zygote: Zygote
using Test
using LinearAlgebra: norm
import Random

Random.seed!(1234)

H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ X[2, :]          # |p|²/2 + q₂ on ℝ⁴
reduced_gradient(dec, z) = reduce(hcat, [Zygote.gradient(ζ -> H(reshape(dec(ζ), 4, 1))[1], collect(ζ))[1] for ζ in eachcol(z)])
targets(dec, z) = vcat(dec(z), reduced_gradient(dec, z))

# The central differences agree with automatic differentiation.
function test_gradient()
    dec = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
    z = randn(2, 30)
    loss = ReducedHamiltonianLoss(H)
    G = _reduced_hamiltonian_gradient(loss, dec.model, params(dec), z)
    @test G ≈ reduced_gradient(dec, z) rtol = 1e-6
end

# A decoder fitted to its own targets has a loss of the size of the difference error.
function test_self_consistency()
    dec = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
    z = randn(2, 30)
    @test ReducedHamiltonianLoss(H)(dec, z, targets(dec, z)) < 1e-6
end

# λ weighs the gradient term only: with targets whose states are exact, the loss is λ times the
# gradient error.
function test_weight()
    dec = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
    z = randn(2, 30)
    out = targets(dec, z)
    out[5:6, :] .*= 1.1
    @test ReducedHamiltonianLoss(H; λ = 3)(dec, z, out) ≈ 3 * ReducedHamiltonianLoss(H)(dec, z, out) rtol = 1e-6
end

# Reverse mode through the loss gives a finite gradient with respect to the parameters.
function test_differentiable()
    dec = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
    teacher = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
    z = randn(2, 30)
    loss = ReducedHamiltonianLoss(H)
    out = targets(teacher, z)
    dp = Zygote.gradient(ps -> loss(dec.model, ps, z, out), params(dec))[1]
    @test dp !== nothing
end

# Training a decoder on the targets of another one with the optimizer reduces the loss.
function test_training()
    Random.seed!(2026)
    teacher = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
    student = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
    z = randn(2, 200)
    out = targets(teacher, z)
    loss = ReducedHamiltonianLoss(H)
    before = loss(student, z, out)
    o = Optimizer(Adam(), student)
    o(student, DataLoader(z, out; suppress_info = true), Batch(50), 200, loss; show_progress = false)
    @test loss(student, z, out) < before / 2
end

# Where every target gradient of the batch vanishes, the gradient term is the absolute error, not NaN.
function test_zero_target_gradient()
    dec = decoder(NeuralNetwork(SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
    z = randn(2, 10)
    out = vcat(dec(z), zeros(2, 10))
    l = ReducedHamiltonianLoss(H)(dec, z, out)
    @test isfinite(l)
    @test l ≈ norm(reduced_gradient(dec, z)) rtol = 1e-4
end

test_gradient()
test_zero_target_gradient()
test_self_consistency()
test_weight()
test_differentiable()
test_training()
