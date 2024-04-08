using GeometricMachineLearning 
using GeometricMachineLearning: mat_tensor_mul
using LinearAlgebra: tr
using Zygote: pullback
using Test

function triangular_assignment_test(T=Float64, n::Int=5)
    A = rand(T, n, n)
    LT = LowerTriangular(A)
    UT = UpperTriangular(A)

    @test tr(A) ≈ sum(A - LT - UT)
end

function triangular_multiplication_test(T=Float64, n::Int=5)
    Aₗ = rand(LowerTriangular{T}, n)
    Aᵤ = rand(UpperTriangular{T}, n)

    B = rand(T, n, n)
    @test Aₗ * B ≈ Matrix{T}(Aₗ) * B
    @test Aᵤ * B ≈ Matrix{T}(Aᵤ) * B
end

function triangular_tensor_multiplication_test(T=Float64, n::Int=5)
    Aₗ = rand(LowerTriangular{T}, n)
    Aᵤ = rand(UpperTriangular{T}, n)

    B = rand(T, n, n, n)
    AₗB = mat_tensor_mul(Aₗ, B)
    AᵤB = mat_tensor_mul(Aᵤ, B)
    for i in 1:n
        @test AₗB[:, :, i] ≈ Aₗ * B[:, :, i]
        @test AᵤB[:, :, i] ≈ Aᵤ * B[:, :, i]
    end
end

function triangular_tensor_multiplication_pullback_test(T=Float64, n::Int=5)
    Aₗ = rand(LowerTriangular{T}, n)
    Aᵤ = rand(LowerTriangular{T}, n)

    B = rand(T, n, n, n)
    C_diff = rand(T, n, n, n)

    total_pb_lower = pullback(mat_tensor_mul, Aₗ, B)[2](C_diff)
    total_pb_upper = pullback(mat_tensor_mul, Aᵤ, B)[2](C_diff)

    for i in axes(total_pb_lower[2], 3)
        total_pb_lower[2][:, :, i] ≈ pullback(*, Aₗ, B[:, :, i])[2](C_diff[:, :, i])[2]
        total_pb_upper[2][:, :, i] ≈ pullback(*, Aᵤ, B[:, :, i])[2](C_diff[:, :, i])[2]
    end
end

triangular_assignment_test()
triangular_multiplication_test()
triangular_tensor_multiplication_test()
triangular_tensor_multiplication_pullback_test()