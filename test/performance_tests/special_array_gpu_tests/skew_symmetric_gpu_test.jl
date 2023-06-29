import LinearAlgebra
import GeometricMachineLearning
import CUDA
using GPUArrays
using Printf
using Test

function test_skew_symmetric(dev::GeometricMachineLearning.Device, N)
    A = randn(N, N)
    B = randn(N, N)

    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(dev, A)

    A₂ = GeometricMachineLearning.SkewSymMatrix(A) |> map_to_dev
    B₂ = GeometricMachineLearning.SkewSymMatrix(B) |> map_to_dev

    if dev == CUDA.device()
        @test (typeof(A₂ + B₂) <: GeometricMachineLearning.SkewSymMatrix{T, VT} where {T, VT<:AbstractGPUVector{T}})
    end

    @time A₂ + B₂;
end

for N = 1000:1000:5000
    print("N = ", N, "\n")
    @printf "GeometricMachineLearning cpu: "
    test_skew_symmetric(GeometricMachineLearning.CPUDevice(), N)

    @printf "GeometricMachineLearning gpu: "
    test_skew_symmetric(CUDA.device(), N)
    print("\n")
end
