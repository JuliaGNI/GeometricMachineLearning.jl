using GeometricMachineLearning
using GeometricMachineLearning: params, solve!
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems: CoupledHarmonicOscillator
using LinearAlgebra: svd, norm, I
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(123)

basis(nn::NeuralNetwork) = Matrix(params(nn).L1.weight)

# Each guard of the constructor and of `solve!`. `PODArch(N, N)` would build `StiefelLayer{N, N}`
# twice, whose composition `VᵀVᵀ` is not the identity, and `compute_iterations` a range of step zero.
@test_throws r"^ArgumentError: reduced_dim \(0\) must be at least 1\.$" PODArch(4, 0)
@test_throws r"^ArgumentError: full_dim \(4\) must be greater than reduced_dim \(4\)\.$" PODArch(
    4, 4)
@test_throws r"^ArgumentError: full_dim \(3\) must be greater than reduced_dim \(5\)\.$" PODArch(
    3, 5)
@test_throws r"^ArgumentError: input has 4 snapshots, fewer than reduced_dim \(6\)\.$" solve!(
    NeuralNetwork(PODArch(10, 6)), rand(10, 4))

# A snapshot matrix of the wrong number of rows raises the error before `solve!` writes a weight.
let nn = NeuralNetwork(PODArch(4, 1))
    W₀ = basis(nn)
    @test_throws r"^ArgumentError: input has 1 rows, not full_dim \(4\)\.$" solve!(nn, ones(1, 3))
    @test basis(nn) == W₀
end

# POD is the best rank-n linear projection, so its error is the norm of the discarded singular
# values, and the encoder and the decoder hold the same basis.
function test_error_is_optimal(::Type{T}, N::Integer, n::Integer) where {T}
    M = rand(T, N, 10 * N)
    pod_nn = NeuralNetwork(PODArch(N, n), T)
    pod_error = solve!(pod_nn, M)

    σ = svd(M).S
    @test pod_error isa T
    @test pod_error ≈ norm(σ[(n + 1):end]) / norm(M) rtol = 100 * eps(T)
    @test Matrix(params(pod_nn).L2.weight) ≈ basis(pod_nn) rtol = 8 * eps(T)
end

# Every other input is the snapshot matrix in another shape: a 3-tensor is its columns side by side,
# a `DataLoader` holds it, and a `(q, p)` `DataLoader` holds its upper and lower halves.
function test_input_methods(::Type{T}, N::Integer, n::Integer) where {T}
    q, p = rand(T, N ÷ 2, 6, 5), rand(T, N ÷ 2, 6, 5)
    M = reshape(vcat(q, p), N, 30)
    reference_nn = NeuralNetwork(PODArch(N, n), T)
    reference_error = solve!(reference_nn, M)

    inputs = (("3-tensor", vcat(q, p)),
        ("DataLoader", DataLoader(M; autoencoder = true, suppress_info = true)),
        ("(q, p) DataLoader",
            DataLoader((q = reshape(q, N ÷ 2, 30), p = reshape(p, N ÷ 2, 30));
                autoencoder = true, suppress_info = true)))
    for (name, input) in inputs
        @testset "$name" begin
            pod_nn = NeuralNetwork(PODArch(N, n), T)
            @test solve!(pod_nn, input) ≈ reference_error rtol = 8 * eps(T)
            @test basis(pod_nn) ≈ basis(reference_nn) rtol = 8 * eps(T)
        end
    end
end

function test_encoder_and_decoder(::Type{T}, N::Integer, n::Integer) where {T}
    pod_nn = NeuralNetwork(PODArch(N, n), T)
    solve!(pod_nn, rand(T, N, 10 * N))
    pod_encoder = encoder(pod_nn)
    pod_decoder = decoder(pod_nn)

    test_vec = rand(T, N)
    test_mat = rand(T, N, N)

    @test eltype(pod_nn(test_vec)) === T
    @test pod_nn(test_vec) ≈ pod_decoder(pod_encoder(test_vec))
    @test pod_nn(test_mat) ≈ pod_decoder(pod_encoder(test_mat))
    # the decoder has orthonormal columns, so encoding a decoded vector gives it back
    test_reduced = rand(T, n)
    @test pod_encoder(pod_decoder(test_reduced)) ≈ test_reduced
end

# PSD is a rank-n linear projection too, so POD can only be better.
function test_error_not_above_psd(::Type{T}, N::Integer, n::Integer) where {T}
    M = rand(T, N, 10 * N)
    pod_error = solve!(NeuralNetwork(PODArch(N, n), T), M)
    psd_error = solve!(NeuralNetwork(PSDArch(N, n), T), M)

    @test pod_error ≤ psd_error
end

for T in REAL_ELTYPES
    @testset "$T" begin
        for (N, n) in ((10, 6), (20, 10), (7, 3))
            test_error_is_optimal(T, N, n)
            test_encoder_and_decoder(T, N, n)
        end
        test_input_methods(T, 10, 4)
        test_error_not_above_psd(T, 10, 6)
        test_error_not_above_psd(T, 20, 10)
    end
end

# Two uncoupled oscillators H = Σᵢ pᵢ²/2 + φᵢqᵢ²/2 with φ₁ = 0.05, φ₂ = π, started at (0, 0, 1, 3):
# the largest amplitudes are q₁ (2√5) and p₂ (3), so POD keeps [e₁ e₄], whereas PSD has to keep q and p
# of the same oscillator and picks [e₁ e₃].
function test_two_oscillators()
    parameters = (m₁ = 1.0, m₂ = 1.0, k₁ = 0.05, k₂ = π, k = 0.0)
    problem = CoupledHarmonicOscillator.hodeproblem([0.0, 0.0], [1.0, 3.0];
        timespan = (0.0, 500.0), timestep = 0.05, parameters)
    sol = integrate(problem, ImplicitMidpoint())
    # the solution is indexed from 0; `collect` makes a plain Vector that `hcat` can take
    M = reduce(hcat, collect([vcat(sol.q[i], sol.p[i]) for i in eachindex(sol.q)]))

    pod_nn = NeuralNetwork(PODArch(4, 2))
    pod_error = solve!(pod_nn, M)
    psd_nn = NeuralNetwork(PSDArch(4, 2))
    psd_error = solve!(psd_nn, M)

    @test abs.(basis(pod_nn)) ≈ [1 0; 0 0; 0 0; 0 1] atol = 1e-2
    @test abs.(basis(psd_nn)) ≈ [1; 0;;] atol = 1e-2
    @test pod_error < psd_error
end

test_two_oscillators()
