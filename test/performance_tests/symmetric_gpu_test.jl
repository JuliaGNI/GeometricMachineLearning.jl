import LinearAlgebra
import GeometricMachineLearning
import CUDA
using GPUArrays
using Printf
using Test

gpu(A::AbstractArray) = GeometricMachineLearning.convert_to_gpu(CUDA.device(), A)

function test_symmetric(N)
    A = randn(N, N)
    B = randn(N, N)

    A₁ = LinearAlgebra.Symmetric(A) 
    B₁ = LinearAlgebra.Symmetric(B)

    A₂ = GeometricMachineLearning.SymmetricMatrix(A)    
    B₂ = GeometricMachineLearning.SymmetricMatrix(B)

    A_gpu₁ = A₁ |> gpu 
    B_gpu₁ = B₁ |> gpu

    A_gpu₂ = A₂ |> gpu
    B_gpu₂ = B₂ |> gpu 
    @test (typeof(A_gpu₂ + B_gpu₂) <: GeometricMachineLearning.SymmetricMatrix{T, VT} where {T, VT<:AbstractGPUVector{T}})

    @printf "LinearAlgebra cpu:             " 
    @time A₁ + B₁;

    @printf "LinearAlgebra gpu:             " 
    @time A_gpu₁ + B_gpu₁;

    @printf "GeometricMachineLearning cpu:  " 
    @time A₂ + B₂;

    @printf "GeometricMachineLearning gpu:  " 
    @time A_gpu₂ + B_gpu₂;

end

for N = 1000:1000:5000
    print("N = ", N, "\n")
    test_symmetric(N)
    print("\n")
end
