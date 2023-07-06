using GeometricMachineLearning: tensor_mat_mul, mat_tensor_mul
using ChainRulesTestUtils

test_rrule(tensor_mat_mul, rand(3, 3, 3), rand(3, 3))
test_rrule(mat_tensor_mul, rand(3, 3), rand(3, 3, 3))
#compute the derivative with FiniteDifferences.jl
