import LinearAlgebra
import GeometricMachineLearning
using CUDA
using Printf
using Test


function test_symmetric(N)
    A = randn(N, N)
    B = randn(N, N)

    A₁ = LinearAlgebra.Symmetric(A) 
    B₁ = LinearAlgebra.Symmetric(B)

    A₂ = GeometricMachineLearning.SymmetricMatrix(A)    
    B₂ = GeometricMachineLearning.SymmetricMatrix(B)

    A_cu₁ = A₁ |> cu
    B_cu₁ = B₁ |> cu

    A_cu₂ = A₂ |> cu 
    B_cu₂ = B₂ |> cu 

    @test (typeof(A_cu₂ + B_cu₂) <: GeometricMachineLearning.SymmetricMatrix{T, CuArray{T, 1, CUDA.Mem.DeviceBuffer}} where {T})

    @printf "LinearAlgebra cpu:             " 
    @time A₁ + B₁;

    @printf "LinearAlgebra gpu:             " 
    @time A_cu₁ + B_cu₁;

    @printf "GeometricMachineLearning cpu:  " 
    @time A₂ + B₂;

    @printf "GeometricMachineLearning gpu:  " 
    @time A_cu₂ + B_cu₂;

end

for N = 1000:1000:5000
    print("N = ", N, "\n")
    test_symmetric(N)
    print("\n")
end
