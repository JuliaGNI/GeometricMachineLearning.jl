using GeometricMachineLearning
using CUDA 
using Printf 
using Test 
using GPUArrays

import LinearAlgebra

function test_global_section(dev, T, N, n)
    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(dev, A)

    Y = rand(StiefelManifold{T}, N, n) |> map_to_dev

    A = rand(T, N, n) |> map_to_dev

    A_vec = rgrad(Y, A) 

    @printf "GlobalSection: "
    @time λY = GlobalSection(Y)

    @printf "GlobalTangent: "
    @time B = global_rep(λY, A_vec)

    if dev == CUDA.device()
        @test (typeof(λY.λ) <: LinearAlgebra.QRPackedQ{T, AT} where {T, AT<:AbstractGPUMatrix{T}})
        @test (typeof(B) <: GeometricMachineLearning.StiefelLieAlgHorMatrix{T, GeometricMachineLearning.SkewSymMatrix{T, VT}, AT} where {T, VT <: AbstractGPUVector{T}, AT <: AbstractGPUMatrix{T}})

    end
end

T = Float32
for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    @printf "GeometricMachineLearning cpu:  \n"
    test_global_section(GeometricMachineLearning.CPUDevice(), T, N, n)
    @printf "GeometricMachineLearning gpu:  \n"
    test_global_section(CUDA.device(), T, N, n)
    print("\n")
end