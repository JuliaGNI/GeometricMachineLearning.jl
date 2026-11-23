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

# Each guard of the constructor. `PSDArch(N, N)` would build `PSDLayer{N, N}` twice, whose
# composition is not the identity, and `compute_iterations` a range of step zero.
@test_throws r"^ArgumentError: reduced_dim \(0\) must be at least 1\.$" PSDArch(4, 0)
@test_throws r"^ArgumentError: full_dim \(4\) must be greater than reduced_dim \(4\)\.$" PSDArch(
    4, 4)
@test_throws r"^ArgumentError: full_dim \(4\) must be greater than reduced_dim \(6\)\.$" PSDArch(
    4, 6)
@test_throws r"^ArgumentError: full_dim \(5\) and reduced_dim \(2\) must be even\.$" PSDArch(5, 2)
@test_throws r"^ArgumentError: full_dim \(6\) and reduced_dim \(3\) must be even\.$" PSDArch(6, 3)
# One snapshot gives a cotangent lift of 2 columns, too few for 3 = reduced_dim ÷ 2 basis vectors.
@test_throws r"^ArgumentError: the cotangent lift of input has 2 columns, fewer than reduced_dim ÷ 2 \(3\)\.$" solve!(
    NeuralNetwork(PSDArch(10, 6)), rand(10, 1))

for T in REAL_ELTYPES
    all_tests(T, 10, 6)
    all_tests(T, 20, 10)
end
