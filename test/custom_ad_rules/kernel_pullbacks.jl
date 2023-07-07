using GeometricMachineLearning: tensor_mat_mul, mat_tensor_mul, tensor_tensor_mul
using ChainRulesTestUtils

function main(first_dim, second_dim, third_dim, third_tensor_dim)
    test_rrule(tensor_mat_mul, rand(first_dim, second_dim, third_tensor_dim), rand(second_dim, third_dim))
    test_rrule(mat_tensor_mul, rand(first_dim, second_dim), rand(second_dim, third_dim, third_tensor_dim))
    test_rrule(tensor_tensor_mul, rand(first_dim, second_dim, third_tensor_dim), rand(second_dim, third_dim, third_tensor_dim))
    #compute the derivative with FiniteDifferences.jl
end

dim_range = 10
num_tests = 10
for _ in 1:num_tests
    first_dim = Int(ceil(dim_range*rand()))
    second_dim = Int(ceil(dim_range*rand()))
    third_dim = Int(ceil(dim_range*rand()))
    third_tensor_dim = Int(ceil(dim_range*rand()))
    print("dims are : (", first_dim, ", ", second_dim, ", ", third_dim, ")\n")
    main(first_dim, second_dim, third_dim, third_tensor_dim)
end