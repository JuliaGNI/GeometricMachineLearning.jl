using GeometricMachineLearning: tensor_mat_mul, mat_tensor_mul, tensor_tensor_mul, tensor_transpose_tensor_mul
using ChainRulesTestUtils
using Printf

function main(first_dim, second_dim, third_dim, third_tensor_dim)
    test_rrule(tensor_mat_mul, rand(first_dim, second_dim, third_tensor_dim), rand(second_dim, third_dim))
    test_rrule(mat_tensor_mul, rand(first_dim, second_dim), rand(second_dim, third_dim, third_tensor_dim))
    test_rrule(tensor_tensor_mul, rand(first_dim, second_dim, third_tensor_dim), rand(second_dim, third_dim, third_tensor_dim))
    test_rrule(tensor_transpose_tensor_mul, rand(second_dim, first_dim, third_tensor_dim), rand(second_dim, third_dim, third_tensor_dim))
    #compute the derivative with FiniteDifferences.jl
end

const dim_range = 10
const num_tests = 10
function test(verbose=false)
    for _ in 1:num_tests
        first_dim = Int(ceil(dim_range*rand()))
        second_dim = Int(ceil(dim_range*rand()))
        third_dim = Int(ceil(dim_range*rand()))
        third_tensor_dim = Int(ceil(dim_range*rand()))
        verbose ? printn("dims are : (", first_dim, ", ", second_dim, ", ", third_dim, ", ", third_tensor_dim, ")") : nothing
        main(first_dim, second_dim, third_dim, third_tensor_dim)
        verbose ? printn() : nothing
    end
end