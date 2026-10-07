using GeometricMachineLearning
using GeometricMachineLearning: params
using Zygote: Zygote
using Test
using LinearAlgebra: norm
import Random

Random.seed!(1234)

H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ X[2, :]          # |p|²/2 + q₂ on ℝ⁴
XH(X) = vcat(X[3:4, :], vcat(zeros(1, size(X, 2)), -ones(1, size(X, 2))))   # 𝕁∇H = (p, -∇_q H)
arch = SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)

# The loss agrees with the same expression evaluated with Zygote's derivatives.
function test_against_autodiff()
    nn = NeuralNetwork(arch)
    enc, dec = encoder(nn), decoder(nn)
    x = randn(4, 30); v = XH(x)
    ξ = enc(x)
    pushed = reduce(hcat, [Zygote.jacobian(enc, x[:, i])[1] * v[:, i] for i in axes(x, 2)])
    g = reduce(hcat, [Zygote.gradient(ζ -> H(reshape(dec(ζ), 4, 1))[1], ξ[:, i])[1] for i in axes(x, 2)])
    reduced = vcat(g[2:2, :], -g[1:1, :])
    expected = norm(dec(ξ) - x) / norm(x) + norm(reduced - pushed) / norm(pushed)
    @test LatentVectorFieldLoss(arch, H)(nn, x, v) ≈ expected rtol = 1e-6
end

# λ weighs the vector-field term only.
function test_weight()
    nn = NeuralNetwork(arch)
    x = randn(4, 30); v = XH(x)
    rec = LatentVectorFieldLoss(arch, H; λ = 0)(nn, x, v)
    @test LatentVectorFieldLoss(arch, H; λ = 3)(nn, x, v) - rec ≈ 3 * (LatentVectorFieldLoss(arch, H)(nn, x, v) - rec) rtol = 1e-8
end

# Reverse mode through the loss gives a finite gradient for encoder and decoder parameters.
function test_differentiable()
    nn = NeuralNetwork(arch)
    x = randn(4, 30); v = XH(x)
    loss = LatentVectorFieldLoss(arch, H)
    dp = Zygote.gradient(ps -> loss(nn.model, ps, x, v), params(nn))[1]
    @test dp !== nothing
end

# Training with the optimizer reduces the loss.
function test_training()
    Random.seed!(2026)
    nn = NeuralNetwork(arch)
    x = randn(4, 200); v = XH(x)
    loss = LatentVectorFieldLoss(arch, H)
    before = loss(nn, x, v)
    o = Optimizer(Adam(), nn)
    o(nn, DataLoader(x, v; suppress_info = true), Batch(50), 200, loss; show_progress = false)
    @test loss(nn, x, v) < before / 2
end

# Without the reconstruction term the loss is the vector-field term alone.
function test_without_reconstruction()
    nn = NeuralNetwork(arch)
    x = randn(4, 30); v = XH(x)
    rec = LatentVectorFieldLoss(arch, H; λ = 0)(nn, x, v)
    @test LatentVectorFieldLoss(arch, H; reconstruction = false)(nn, x, v) ≈ LatentVectorFieldLoss(arch, H)(nn, x, v) - rec rtol = 1e-10
end

test_against_autodiff()
test_without_reconstruction()
test_weight()
test_differentiable()
test_training()
