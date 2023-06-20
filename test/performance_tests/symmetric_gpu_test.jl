import LinearAlgebra
import GeometricMachineLearning
import CUDA
using GPUArrays
using Printf
using Test

function test_symmetric(dev::GeometricMachineLearning.Device, N)
    A = randn(N, N)
    B = randn(N, N)

    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(dev, A)

    A₂ = GeometricMachineLearning.SymmetricMatrix(A) |> map_to_dev
    B₂ = GeometricMachineLearning.SymmetricMatrix(B) |> map_to_dev

    if dev == CUDA.device()
        @test (typeof(A₂ + B₂) <: GeometricMachineLearning.SymmetricMatrix{T, VT} where {T, VT<:AbstractGPUVector{T}})
    end

    @time A₂ + B₂;
end

for N = 1000:1000:5000
    print("N = ", N, "\n")
    @printf "GeometricMachineLearning cpu: "
    test_symmetric(GeometricMachineLearning.CPUDevice(), N)

    @printf "GeometricMachineLearning gpu: "
    test_symmetric(CUDA.device(), N)
    print("\n")
end
