import LinearAlgebra
import GeometricMachineLearning
using CUDA
using Printf
using Test


function test_skew_symmetric(N)
    A = randn(N, N)
    B = randn(N, N)

    A₂ = GeometricMachineLearning.SkewSymMatrix(A)    
    B₂ = GeometricMachineLearning.SkewSymMatrix(B)

    A_cu₂ = A₂ |> cu 
    B_cu₂ = B₂ |> cu 

    @test (typeof(A_cu₂ + B_cu₂) <: GeometricMachineLearning.SkewSymMatrix{T, CuArray{T, 1, CUDA.Mem.DeviceBuffer}} where {T})

    @printf "GeometricMachineLearning cpu:  " 
    @time A₂ + B₂;

    @printf "GeometricMachineLearning gpu:  " 
    @time A_cu₂ + B_cu₂;

end

for N = 1000:1000:5000
    print("N = ", N, "\n")
    test_skew_symmetric(N)
    print("\n")
end
