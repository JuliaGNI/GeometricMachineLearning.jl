import LinearAlgebra
import GeometricMachineLearning
using CUDA
using Printf
using Test


function test_symmetric(N)
    A = randn(N, N)
    B = randn(N, N)

    A_cu = A |> cu
    B_cu = B |> cu

    A₁ = LinearAlgebra.Symmetric(A) 
    B₁ = LinearAlgebra.Symmetric(B)

    A₂ = GeometricMachineLearning.SymmetricMatrix(A)    
    B₂ = GeometricMachineLearning.SymmetricMatrix(B)

    A_cu₁ = LinearAlgebra.Symmetric(A_cu) 
    B_cu₁ = LinearAlgebra.Symmetric(B_cu)

    A_cu₂ = GeometricMachineLearning.SymmetricMatrix(A_cu)
    B_cu₂ = GeometricMachineLearning.SymmetricMatrix(B_cu)

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
