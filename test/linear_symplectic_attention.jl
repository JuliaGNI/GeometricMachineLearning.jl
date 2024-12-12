using GeometricMachineLearning
using Test
import Random 

Random.seed!(123)

function test_application_of_lsa(n::Integer=4, seq_length::Integer=5, T=Float64)
    l₁ = LinearSymplecticAttentionQ(n, seq_length)
    l₂ = LinearSymplecticAttentionP(n, seq_length)
    ps₁ = NeuralNetwork(l₁, CPU(), T).params
    ps₂ = NeuralNetwork(l₂, CPU(), T).params

    # test for NamedTuple as input
    nt = (q = rand(T, n, seq_length), p = rand(T, n, seq_length))
    @test l₁(nt, ps₁).q ≈ nt.q + nt.p * ps₁.A
    @test l₂(nt, ps₂).p ≈ nt.p + nt.q * ps₂.A

    # test for Array as input
    arr = rand(T, 2 * n, seq_length)
    @test l₁(arr, ps₁) ≈ vcat(arr[1:n, :] + arr[(n + 1):(2 * n), :] * ps₁.A, arr[(n + 1):(2 * n), :])
    @test l₂(arr, ps₂) ≈ vcat(arr[1:n, :], arr[(n + 1):(2 * n), :] + arr[1:n, :] * ps₂.A)
end

test_application_of_lsa()