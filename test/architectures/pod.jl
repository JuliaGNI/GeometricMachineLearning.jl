using GeometricMachineLearning
using GeometricMachineLearning: params
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems: CoupledHarmonicOscillator
using LinearAlgebra: svd, norm, I
using Test
import Random

Random.seed!(123)

# POD is the best rank-n linear projection, so its error is the norm of the discarded singular values.
function test_error_is_optimal(N::Integer, n::Integer)
    M = rand(N, 10 * N)
    pod_nn = NeuralNetwork(PODArch(N, n))
    pod_error = solve!(pod_nn, M)

    σ = svd(M).S
    @test pod_error ≈ norm(σ[(n + 1):end]) / norm(M)
end

function test_encoder_and_decoder(N::Integer, n::Integer)
    pod_nn = NeuralNetwork(PODArch(N, n))
    solve!(pod_nn, DataLoader(rand(N, 10 * N); autoencoder = true))
    pod_encoder = encoder(pod_nn)
    pod_decoder = decoder(pod_nn)

    test_vec = rand(N)
    test_mat = rand(N, N)

    @test pod_nn(test_vec) ≈ pod_decoder(pod_encoder(test_vec))
    @test pod_nn(test_mat) ≈ pod_decoder(pod_encoder(test_mat))
    # the decoder has orthonormal columns, so encoding a decoded vector gives it back
    test_reduced = rand(n)
    @test pod_encoder(pod_decoder(test_reduced)) ≈ test_reduced
end

# PSD is a rank-n linear projection too, so POD can only be better.
function test_error_not_above_psd(N::Integer, n::Integer)
    dl = DataLoader(rand(N, 10 * N); autoencoder = true)
    pod_error = solve!(NeuralNetwork(PODArch(N, n)), dl)
    psd_error = solve!(NeuralNetwork(PSDArch(N, n)), dl)

    @test pod_error ≤ psd_error
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

    @test abs.(Matrix(params(pod_nn).L1.weight)) ≈ [1 0; 0 0; 0 0; 0 1] atol = 1e-2
    @test abs.(Matrix(params(psd_nn).L1.weight)) ≈ [1; 0;;] atol = 1e-2
    @test pod_error < psd_error
end

for (N, n) in ((10, 6), (20, 10), (7, 3))
    test_error_is_optimal(N, n)
    test_encoder_and_decoder(N, n)
end
test_error_not_above_psd(10, 6)
test_error_not_above_psd(20, 10)
test_two_oscillators()
