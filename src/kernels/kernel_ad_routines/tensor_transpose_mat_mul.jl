# This implements the custom pullback for tensor_transpose_mat_mul
function ChainRulesCore.rrule(::typeof(tensor_transpose_mat_mul),
        A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where {T}
    @assert axes(A, 1) == axes(B, 1)
    C = tensor_transpose_mat_mul(A, B)
    function tensor_transpose_mat_mul_pullback(C_diff)
        f̄ = NoTangent()
        A_diff = @thunk mat_tensor_transpose_mul(B, unthunk(C_diff))
        B_diff = @thunk _matrix_cotangent(B, sum(tensor_tensor_mul(A, unthunk(C_diff)), dims = 3))
        return f̄, A_diff, B_diff
    end
    return C, tensor_transpose_mat_mul_pullback
end

mat_tensor_transpose_mul(B, C) = mat_tensor_mul(B, tensor_transpose(C))
