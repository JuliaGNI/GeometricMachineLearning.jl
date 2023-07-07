"""
This implements the custom pullback tor tensor_transpose_tensor_mul
"""

#the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(tensor_transpose_tensor_mul), A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    @assert axes(A, 1) == axes(B, 1)
    C = tensor_transpose_tensor_mul(A, B)
    function tensor_transpose_tensor_mul_pullback(C_diff)
        f̄ = NoTangent()
        #tensor_transpose_mat_mul
        A_diff = @thunk tensor_transpose_tensor_transpose_mul(C_diff, B)
        B_diff = @thunk tensor_tensor_mul(A, C_diff)
        return f̄, A_diff, B_diff
    end
    return C, tensor_transpose_tensor_mul_pullback
end
   
