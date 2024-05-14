using GeometricMachineLearning, Test
using LinearAlgebra: det
import Random 

Random.seed!(1234)

function volume_preserving_attention_tests(N, T=Float32)
    model₁ = VolumePreservingAttention(N, N, skew_sym = false)
    model₂ = VolumePreservingAttention(N, N, skew_sym = true)

    ps₁ = initialparameters(model₁, CPU(), T)
    ps₂ = initialparameters(model₂, CPU(), T)
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

function check_all(T)
    # this checks the cpu version
    volume_preserving_attention_tests(10, T)

    # this checks the "gpu versions"
    volume_preserving_attention_tests(2, T)
    volume_preserving_attention_tests(3, T)
    volume_preserving_attention_tests(4, T)
end

check_all(Float16)
check_all(Float32)
check_all(Float64)