import LinearAlgebra
import GeometricMachineLearning
import CUDA
using GPUArrays
using Printf
using Test

gpu(A::AbstractArray) = GeometricMachineLearning.convert_to_gpu(CUDA.device(), A)

function test_skew_symmetric(N)
    A = randn(N, N)
    B = randn(N, N)

    A₂ = GeometricMachineLearning.SkewSymMatrix(A)    
    B₂ = GeometricMachineLearning.SkewSymMatrix(B)

    A_gpu₂ = A₂ |> gpu 
    B_gpu₂ = B₂ |> gpu 

    @test (typeof(A_gpu₂ + B_gpu₂) <: GeometricMachineLearning.SkewSymMatrix{T, VT} where {T, VT<:AbstractGPUVector{T}})

    @printf "GeometricMachineLearning cpu:  " 
    @time A₂ + B₂;

    @printf "GeometricMachineLearning gpu:  " 
    @time A_gpu₂ + B_gpu₂;

end

for N = 1000:1000:5000
    print("N = ", N, "\n")
    test_skew_symmetric(N)
    print("\n")
end
