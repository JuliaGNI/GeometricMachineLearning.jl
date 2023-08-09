"""
This implements the custom pullback tor tensor_transpose
"""

#the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(tensor_transpose), A::AbstractArray{T, 3}) where T
    C = tensor_transpose(A)
    function tensor_transpose_pullback(C_diff)
        f̄ = NoTangent()
        A_diff = @thunk tensor_transpose(C_diff)
        return f̄, A_diff
    end
    return C, tensor_transpose_pullback
end