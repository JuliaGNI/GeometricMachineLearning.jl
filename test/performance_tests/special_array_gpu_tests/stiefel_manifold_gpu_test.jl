import LinearAlgebra
import GeometricMachineLearning
import CUDA
using GPUArrays
using Printf
using Test


function test_stiefel_manifold(dev, T, N, n)
    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(dev, A)

    Y = rand(GeometricMachineLearning.StiefelManifold{T}, N, n) |> map_to_dev

    @printf "StiefelManifold check: " 
    @time GeometricMachineLearning.check(Y)

    A = rand(T, N, n) |> map_to_dev

    @printf "Riemannian gradient:   "
    @time A_vec = GeometricMachineLearning.rgrad(Y, A)

    if dev == CUDA.device()
        @test (typeof(Y) <: GeometricMachineLearning.StiefelManifold{T, AT} where {T, AT <: AbstractGPUMatrix{T}})
        @test (typeof(A_vec) <: AbstractGPUMatrix)
    end
end


T = Float32
for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    @printf "GeometricMachineLearning cpu:  \n"
    test_stiefel_manifold(GeometricMachineLearning.CPUDevice(), T, N, n)
    @printf "GeometricMachineLearning gpu:  \n"
    test_stiefel_manifold(CUDA.device(), T, N, n)
    print("\n")
end
