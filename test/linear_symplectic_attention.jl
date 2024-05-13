using GeometricMachineLearning
using Test
import Random 

Random.seed!(123)

function test_application_of_lsa(n::Integer=4, seq_length::Integer=5, T=Float64)
    l₁ = LinearSymplecticAttentionQ(n, seq_length)
    l₂ = LinearSymplecticAttentionP(n, seq_length)
    ps₁ = initialparameters(l₁, CPU(), T)
    ps₂ = initialparameters(l₂, CPU(), T)

    nt = (q = rand(T, n, seq_length), p = rand(T, n, seq_length))
    @test l₁(nt, ps₁).q ≈ nt.q + nt.p * ps₁.A
    @test l₂(nt, ps₂).p ≈ nt.p + nt.q * ps₂.A
end

test_application_of_lsa()