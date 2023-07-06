using GeometricMachineLearning: tensor_mat_mul
using ChainRulesTestUtils
test_rrule(tensor_mat_mul, rand(3, 3, 3), rand(3, 3))
#compute the derivative with FiniteDifferences.jl
