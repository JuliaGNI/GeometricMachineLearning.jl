using GeometricMachineLearning
using GeometricMachineLearning: solve!
using Zygote: jacobian
using Test
import Random

include("../helpers/poisson.jl")
include("../helpers/eltypes.jl")

Random.seed!(123)

function test_accuracy(::Type{T}, N::Integer, n::Integer; tol::Real = 0.35) where {T}
    dl = DataLoader(rand(T, N, 10 * N); autoencoder = true)

    psd_nn = NeuralNetwork(PSDArch(N, n), T)
    psd_error = solve!(psd_nn, dl)

    @test eltype(psd_error) === T
    @test psd_error < tol
end

function test_encoder_and_decoder(::Type{T}, N::Integer, n::Integer) where {T}
    psd_nn = NeuralNetwork(PSDArch(N, n), T)
    psd_encoder = encoder(psd_nn)
    psd_decoder = decoder(psd_nn)

    test_vec = rand(T, N)
    test_mat = rand(T, N, N)

    @test eltype(psd_nn(test_vec)) === T
    @test psd_nn(test_vec) ≈ psd_decoder(psd_encoder(test_vec))
    @test psd_nn(test_mat) ≈ psd_decoder(psd_encoder(test_mat))
end

function test_symplecticity(::Type{T}, N::Integer, n::Integer) where {T}
    psd_nn = NeuralNetwork(PSDArch(N, n), T)
    psd_decoder = decoder(psd_nn)
    test_vector = rand(T, n)

    # this matrix should be symplectic
    sympl_mat = jacobian(vec -> psd_decoder(vec), test_vector)[1]
    @test eltype(sympl_mat) === T
    @test eltype(sympl_mat' * poisson_matrix(N, T) * sympl_mat) === T
    @test poisson_matrix(n, T) ≈ sympl_mat' * poisson_matrix(N, T) * sympl_mat

    # test if it's still symplectic after computing the PSD basis
    dl = DataLoader(rand(T, N, 10 * N); autoencoder = true)
    solve!(psd_nn, dl)
    sympl_mat = jacobian(vec -> psd_decoder(vec), test_vector)[1]
    @test poisson_matrix(n, T) ≈ sympl_mat' * poisson_matrix(N, T) * sympl_mat
end

function all_tests(::Type{T}, N::Integer, n::Integer; kwargs...) where {T}
    test_accuracy(T, N, n; kwargs...)
    test_encoder_and_decoder(T, N, n; kwargs...)
    test_symplecticity(T, N, n)
end

for T in REAL_ELTYPES
    all_tests(T, 10, 6)
    all_tests(T, 20, 10)
end
