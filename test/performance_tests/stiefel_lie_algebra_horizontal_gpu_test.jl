import LinearAlgebra
import GeometricMachineLearning
import CUDA
using GPUArrays
using Printf
using Test

function test_stiefel_lie_algebra_horizontal(dev, N, n)
    A = randn(N, N)
    B = randn(N, N)

    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(dev, A)

    A₂ = GeometricMachineLearning.StiefelLieAlgHorMatrix(A, n) |> map_to_dev
    B₂ = GeometricMachineLearning.StiefelLieAlgHorMatrix(B, n) |> map_to_dev

    if dev == CUDA.device()
        @test (typeof(A₂ + B₂) <: GeometricMachineLearning.StiefelLieAlgHorMatrix{T, GeometricMachineLearning.SkewSymMatrix{T, VT}, AT} where {T, VT <: AbstractGPUVector{T}, AT <: AbstractGPUMatrix{T}})
    end

    @time (A₂ + B₂);
end

for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    @printf "GeometricMachineLearning cpu:  " 
    test_stiefel_lie_algebra_horizontal(GeometricMachineLearning.CPUDevice(), N, n)
    @printf "GeometricMachineLearning gpu:  " 
    test_stiefel_lie_algebra_horizontal(CUDA.device(), N, n)
    print("\n")
end
