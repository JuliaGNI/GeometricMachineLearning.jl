import LinearAlgebra
import GeometricMachineLearning
import CUDA
using GPUArrays
using Printf
using Test

gpu(A::AbstractArray) = GeometricMachineLearning.convert_to_gpu(CUDA.device(), A)

function test_stiefel_lie_algebra_horizontal(N, n)
    A = randn(N, N)
    B = randn(N, N)

    A₂ = GeometricMachineLearning.StiefelLieAlgHorMatrix(A, n)    
    B₂ = GeometricMachineLearning.StiefelLieAlgHorMatrix(B, n)

    A_gpu₂ = A₂ |> gpu 
    B_gpu₂ = B₂ |> gpu 

    @test (typeof(A_gpu₂ + B_gpu₂) <: GeometricMachineLearning.StiefelLieAlgHorMatrix{T, GeometricMachineLearning.SkewSymMatrix{T, VT}, AT} where {T, VT <: AbstractGPUVector{T}, AT <: AbstractGPUMatrix{T}})

    @printf "GeometricMachineLearning cpu:  " 
    @time A₂ + B₂;

    @printf "GeometricMachineLearning gpu:  " 
    @time A_gpu₂ + B_gpu₂;

end

for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    test_stiefel_lie_algebra_horizontal(N, n)
    print("\n")
end
