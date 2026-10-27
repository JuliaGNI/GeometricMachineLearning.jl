using GeometricMachineLearning
using GeometricMachineLearning: _mul
using Zygote: pullback
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

# What the triangular types *are* — their storage layout, their arithmetic, their multiplication
# against a dense matrix — is tested in GeometricOptimizers, which defines them
# (`test/special_matrices/triangular.jl` there). What is left here is GML's: batching them over the
# third axis of a tensor with `_mul`, and the pullback of that product.

function triangular_tensor_multiplication_test(::Type{T}, n::Int = 5) where {T}
    Aₗ = rand(StrictlyLowerTriangular{T}, n)
    Aᵤ = rand(StrictlyUpperTriangular{T}, n)

    B = rand(T, n, n, n)
    AₗB = _mul(Aₗ, B)
    AᵤB = _mul(Aᵤ, B)
    @test eltype(AₗB) === T
    @test eltype(AᵤB) === T
    for i in 1:n
        @test AₗB[:, :, i] ≈ Aₗ * B[:, :, i]
        @test AᵤB[:, :, i] ≈ Aᵤ * B[:, :, i]
    end
end

function triangular_tensor_multiplication_pullback_test(::Type{T}, n::Int = 5) where {T}
    Aₗ = rand(StrictlyLowerTriangular{T}, n)
    Aᵤ = rand(StrictlyUpperTriangular{T}, n)

    B = rand(T, n, n, n)
    C_diff = rand(T, n, n, n)

    total_pb_lower = pullback(_mul, Aₗ, B)[2](C_diff)
    total_pb_upper = pullback(_mul, Aᵤ, B)[2](C_diff)

    # The cotangent of the data carries the element type of the seed, as the product does.
    @test eltype(total_pb_lower[2]) === T
    @test eltype(total_pb_upper[2]) === T

    # The batched pullback has to agree slice by slice with the pullback of the single-slice
    # product. These were bare expressions and not `@test`s before, so the loop asserted nothing.
    for i in axes(total_pb_lower[2], 3)
        @test total_pb_lower[2][:, :, i] ≈
              pullback(*, Aₗ, B[:, :, i])[2](C_diff[:, :, i])[2]
        @test total_pb_upper[2][:, :, i] ≈
              pullback(*, Aᵤ, B[:, :, i])[2](C_diff[:, :, i])[2]
    end
end

for T in REAL_ELTYPES
    triangular_tensor_multiplication_test(T)
    triangular_tensor_multiplication_pullback_test(T)
end
