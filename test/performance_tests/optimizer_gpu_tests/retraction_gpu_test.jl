using GeometricMachineLearning
using GeometricMachineLearning: geodesic
using Printf 
using Test 
using GPUArrays
using CUDA 

function test_retraction(dev, T, N, n)
    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(dev, A)

    B = rand(StiefelLieAlgHorMatrix{T}, N, n) |> map_to_dev
    B = B |> map_to_dev

    @time Y = geodesic(B)

    if dev == CUDA.device()
        @test (typeof(Y) <: StiefelManifold{T, AT} where {T, AT <: AbstractGPUMatrix})
    end
end

T = Float32
for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    @printf "GeometricMachineLearning cpu: "
    test_retraction(GeometricMachineLearning.CPUDevice(), T, N, n)

    @printf "GeometricMachineLearning gpu: "
    test_retraction(CUDA.device(), T, N, n)
    print("\n")
end