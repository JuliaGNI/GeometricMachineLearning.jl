using GeometricMachineLearning
using GeometricMachineLearning: _mul, tensor_mat_skew_sym_assign,
                                tensor_transpose_tensor_mul
using Test
import Random

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

function test_first_option(n::Int = 10, seq_length::Int = 10, batch_size::Int = 20)
    A = rand(SkewSymMatrix, n)
    x = rand(n, seq_length, batch_size)
    xAx = tensor_transpose_tensor_mul(x, _mul(A, x))
    isskew(xAx)
end

function test_second_option(n::Int = 10, seq_length::Int = 10, batch_size::Int = 20)
    A = rand(n, n)
    x = rand(n, seq_length, batch_size)
    xAx = tensor_mat_skew_sym_assign(x, A) / √n
    isskew(xAx - permutedims(xAx, (2, 1, 3)))
end

test_first_option()
test_second_option()
