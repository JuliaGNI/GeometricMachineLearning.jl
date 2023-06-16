using GeometricMachineLearning
using GeometricMachineLearning: geodesic
using Printf 
using Test 
using GPUArrays
using CUDA 

gpu(A::AbstractArray) = GeometricMachineLearning.convert_to_gpu(CUDA.device(), A)

function test_retraction(T, N, n)
    B = rand(StiefelLieAlgHorMatrix{T}, N, n)
    B_gpu = B |> gpu

    @printf "cpu geodesic retraction:  "
    @time Y = geodesic(B)

    @printf "gpu geodesic retraction:  "
    @time Y_gpu = geodesic(B_gpu)

    @test typeof(Y_gpu <: StiefelManifold{T, AT} where {T, AT <: AbstractGPUMatrix})
end

T = Float32
for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    test_retraction(T, N, n)
    print("\n")
end