"""
This implements the custom pullback for tensor_transpose_mat_mul
"""
function ChainRulesCore.rrule(::typeof(tensor_transpose_mat_mul), A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where T
    @assert axes(A, 2) == axes(B, 1)
    C = tensor_transpose_mat_mul(A, B)
    function tensor_transpose_mat_mul_pullback(C_diff)
        f̄ = NoTangent()
        #tensor_transpose_mat_mul
        A_diff = @thunk tensor_transpose_mat_mul(C_diff, B')
        B_diff = @thunk sum(tensor_tensor_mul(A, C_diff), dims=3)
        return f̄, A_diff, B_diff
    end
    return C, tensor_transpose_mat_mul_pullback
end

tensor_transpose_mat_mul(A::Thunk, B::AbstractMatrix) = Thunk(() -> tensor_transpose_mat_mul(unthunk(A), B))

#function tensor_transpose_tensor_mul(A::AbstractArray{T, 3}, B::Thunk) where T 
#    Thunk(() -> tensor_transpose_tensor_mul(A, unthunk(B)))
#end
