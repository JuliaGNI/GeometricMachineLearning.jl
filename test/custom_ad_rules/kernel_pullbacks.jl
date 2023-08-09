using GeometricMachineLearning: tensor_mat_mul, mat_tensor_mul, tensor_tensor_mul, tensor_transpose_tensor_mul, assign_q_and_p
using ChainRulesTestUtils
using Printf

const verbose = false

function main(first_dim, second_dim, third_dim, third_tensor_dim)
    # the first two are tests for the splitting x -> (q, p) (vector and matrix)
    test_rrule(assign_q_and_p, rand(first_dim*2), first_dim)
    test_rrule(assign_q_and_p, rand(first_dim*2, second_dim), first_dim)
    test_rrule(assign_q_and_p, rand(first_dim*2, second_dim, third_dim), first_dim)
    test_rrule(tensor_mat_mul, rand(first_dim, second_dim, third_tensor_dim), rand(second_dim, third_dim))
    test_rrule(mat_tensor_mul, rand(first_dim, second_dim), rand(second_dim, third_dim, third_tensor_dim))
    test_rrule(tensor_tensor_mul, rand(first_dim, second_dim, third_tensor_dim), rand(second_dim, third_dim, third_tensor_dim))
    test_rrule(tensor_transpose_tensor_mul, rand(second_dim, first_dim, third_tensor_dim), rand(second_dim, third_dim, third_tensor_dim))
    test_rrule(tensor_transpose, rand(first_dim, secon_dim, third_tensor_dim))
    test_rrule(tensor_exponential, rand(first_dim, first_dim, third_tensor_dim))
    #compute the derivative with FiniteDifferences.jl
end

const dim_range = 10
const num_tests = 10
for _ in 1:num_tests
    first_dim = Int(ceil(dim_range*rand()))
    second_dim = Int(ceil(dim_range*rand()))
    third_dim = Int(ceil(dim_range*rand()))
    third_tensor_dim = Int(ceil(dim_range*rand()))
    verbose && println("dims are : (", first_dim, ", ", second_dim, ", ", third_dim, ", ", third_tensor_dim, ")")
    main(first_dim, second_dim, third_dim, third_tensor_dim)
    verbose && println()
end