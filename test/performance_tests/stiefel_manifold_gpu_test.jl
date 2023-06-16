import LinearAlgebra
import GeometricMachineLearning
import CUDA
using GPUArrays
using Printf
using Test

#specify device
gpu(A::AbstractArray) = GeometricMachineLearning.convert_to_gpu(CUDA.device(), A)

function test_stiefel_manifold(T, N, n)
    Y = rand(GeometricMachineLearning.StiefelManifold{T}, N, n)

    Y_gpu = Y |> gpu 

    @test (typeof(Y_gpu) <: GeometricMachineLearning.StiefelManifold{T, AT} where {T, AT <: AbstractGPUMatrix{T}})

    @printf "StiefelManifold cpu check:  " 
    @time GeometricMachineLearning.check(Y)

    @printf "StiefelManifold gpu check:  " 
    @time GeometricMachineLearning.check(Y_gpu)

    A = rand(T, N, n)
    A_gpu = A |> gpu 

    @printf "Riemannian gradient cpu:   "
    @time A_vec = GeometricMachineLearning.rgrad(Y, A)

    @printf "Riemannian gradient gpu:   "
    @time A_vec_gpu = GeometricMachineLearning.rgrad(Y_gpu, A_gpu)

    @test (typeof(A_vec_gpu) <: AbstractGPUMatrix)
end


T = Float32
for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    test_stiefel_manifold(T, N, n)
    print("\n")
end
