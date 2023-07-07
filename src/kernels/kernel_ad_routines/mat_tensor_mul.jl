"""
This implements the custom pullback tor mat_tensor_mul
"""

#the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(mat_tensor_mul), B::AbstractMatrix{T}, A::AbstractArray{T, 3}) where T
    @assert axes(A, 1) == axes(B, 2)
    C = mat_tensor_mul(B, A)
    function mat_tensor_mul_pullback(C_diff)
        f̄ = NoTangent()
        #tensor_transpose_mat_mul
        B_diff = @thunk sum(tensor_tensor_transpose_mul(C_diff, A), dims=3)
        A_diff = @thunk mat_tensor_mul(B', C_diff)
        return f̄, B_diff, A_diff
    end
    return C, mat_tensor_mul_pullback
end
   
mat_tensor_mul(B::AbstractMatrix, A::Thunk) = Thunk(() -> mat_tensor_mul(B, unthunk(A)))
    
function tensor_tensor_transpose_mul(B::Thunk, A::AbstractArray{T, 3}) where T 
    Thunk(() -> tensor_tensor_transpose_mul(unthunk(B), A))
end

