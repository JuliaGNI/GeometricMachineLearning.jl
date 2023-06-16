using GeometricMachineLearning
using CUDA 
using Printf 
using Test 
using GPUArrays

import LinearAlgebra

#specify device
gpu(A::AbstractArray) = GeometricMachineLearning.convert_to_gpu(CUDA.device(), A)

function test_global_section(T, N, n)
    Y = rand(StiefelManifold{T}, N, n)
    Y_gpu = Y |> gpu 

    A = rand(T, N, n)
    A_gpu = A |> gpu 

    A_vec = rgrad(Y, A)
    A_gpu_vec = rgrad(Y_gpu, A_gpu)

    @printf "GlobalSection for cpu: "
    @time λY = GlobalSection(Y)
    
    @printf "GlobalSection for gpu: "
    @time λY_gpu = GlobalSection(Y_gpu)

    #result of QR decomposition
    @test (typeof(λY_gpu.λ) <: LinearAlgebra.QRPackedQ{T, AT} where {T, AT<:AbstractGPUMatrix{T}})

    @printf "GlobalTangent for cpu: "
    @time B = global_rep(λY, A_vec)

    @printf "GlobalTangent for gpu: "
    @time B_gpu = global_rep(λY_gpu, A_gpu_vec)

    @test (typeof(B_gpu) <: GeometricMachineLearning.StiefelLieAlgHorMatrix{T, GeometricMachineLearning.SkewSymMatrix{T, VT}, AT} where {T, VT <: AbstractGPUVector{T}, AT <: AbstractGPUMatrix{T}})
end

T = Float32
for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    test_global_section(T, N, n)
    print("\n")
end