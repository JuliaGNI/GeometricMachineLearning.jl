using KernelAbstractions
using GeometricMachineLearning: tensor_mat_mul
using GeometricMachineLearning: tensor_transpose_mat_mul
using GeometricMachineLearning: tensor_transpose_tensor_mul
using GeometricMachineLearning: tensor_tensor_mul
using ChainRulesCore

#the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(tensor_mat_mul), A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where T
    sizeA = size(A); sizeB = size(B)
    @assert sizeA[2] == sizeB[1] 
    C = tensor_mat_mul(A, B)
    function tensor_mat_mul_pullback(C_diff)
        f̄ = NoTangent()
        #tensor_transpose_mat_mul
        A_diff = @thunk tensor_mat_mul(C_diff, B')
        B_diff = @thunk tensor_transpose_tensor_mul(A, C_diff)
        return f̄, A_diff, B_diff
    end
    return C, tensor_mat_mul_pullback
end
    
#rrule is compared to finite differences (FiniteDifferences.jl) 
using ChainRulesTestUtils
test_rrule(tensor_mat_mul, rand(3, 3, 3), rand(3, 3))
