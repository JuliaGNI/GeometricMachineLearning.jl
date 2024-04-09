using GeometricMachineLearning, Test
using LinearAlgebra: det
import Random 

Random.seed!(1234)

function volume_preserving_attention_tests(N, T=Float32)
    model₁ = VolumePreservingAttention(N, skew_sym = false)
    model₂ = VolumePreservingAttention(N, skew_sym = true)

    ps₁ = initialparameters(CPU(), T, model₁)
    ps₂ = initialparameters(CPU(), T, model₂)
    @test typeof(ps₁.A) <: AbstractMatrix{T} 
    @test typeof(ps₂.A) <: SkewSymMatrix{T} 

    # check if the layers are volume preserving
    A = randn(T, N, N)
    det₁ = det(A)
    det₂ = det(model₁(A, ps₁))
    det₃ = det(model₂(A, ps₂))
    @test det₁ ≈ det₂
    @test det₂ ≈ det₃
end

# this checks the cpu version
volume_preserving_attention_tests(10)

# this checks the "gpu versions"
volume_preserving_attention_tests(2)
volume_preserving_attention_tests(3)
volume_preserving_attention_tests(4)
volume_preserving_attention_tests(5)
volume_preserving_attention_tests(6)