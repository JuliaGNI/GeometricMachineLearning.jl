using GeometricMachineLearning
using GeometricMachineLearning: _mul, tensor_mat_skew_sym_assign,
                                tensor_transpose_tensor_mul
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

function isskew(A::AbstractMatrix)
    @test -A ≈ A'
end

function isskew(A::AbstractArray{T, 3}) where {T}
    for i in axes(A, 3)
        A_matrix = @view A[:, :, i]
        isskew(A_matrix)
    end
end

function test_first_option(::Type{T}, n::Int = 10, seq_length::Int = 10,
        batch_size::Int = 20) where {T}
    A = rand(SkewSymMatrix{T}, n)
    x = rand(T, n, seq_length, batch_size)
    xAx = tensor_transpose_tensor_mul(x, _mul(A, x))
    @test eltype(xAx) === T
    isskew(xAx)
end

function test_second_option(::Type{T}, n::Int = 10, seq_length::Int = 10,
        batch_size::Int = 20) where {T}
    A = rand(T, n, n)
    x = rand(T, n, seq_length, batch_size)
    xAx = tensor_mat_skew_sym_assign(x, A) / √T(n)
    @test eltype(xAx) === T
    isskew(xAx - permutedims(xAx, (2, 1, 3)))
end

for T in REAL_ELTYPES
    test_first_option(T)
    test_second_option(T)
end
