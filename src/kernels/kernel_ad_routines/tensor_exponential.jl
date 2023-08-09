"""
This implements the custom pullbacks relevant for tensor_exponential
"""

#the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(assign_matrix_from_tensor), A::AbstractArray{T, 3}) where T
    B = assign_matrix_from_tensor(A)
    function assign_matrix_from_tensor_pullback(B_diff)
        f̄ = NoTangent()
        A_diff = @thunk assign_tensor_from_matrix(B_diff)
        return f̄, A_diff
    end
    return B, assign_matrix_from_tensor_pullback
end

function ChainRulesCore.rrule(::typeof(assign_tensor_from_matrix), B::AbstractMatrix{T}) where T
    A = assign_tensor_from_matrix(B)
    function assign_tensor_from_matrix_pullback(A_diff)
        f̄ = NoTangent()
        B_diff = @thunk assign_matrix_from_tensor(A_diff)
        return f̄, A_diff
    end
    return A, assign_tensor_from_matrix_pullback
end