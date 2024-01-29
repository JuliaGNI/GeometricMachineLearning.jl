"""
This implements the custom pullback tor matrix_transpose
"""
function ChainRulesCore.rrule(::typeof(matrix_transpose), A::AbstractArray{T, 2}) where T
    C = matrix_transpose(A)
    function matrix_transpose_pullback(C_diff)
        f̄ = NoTangent()
        A_diff = @thunk matrix_transpose(C_diff)
        return f̄, A_diff
    end
    return C, matrix_transpose_pullback
end

function matrix_transpose(A::Thunk)
    Thunk(() -> matrix_transpose(unthunk(A)))
end