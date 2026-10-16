function ChainRulesCore.rrule(::typeof(tensor_transpose_tensor_mul),
        A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where {T}
    @assert axes(A, 1) == axes(B, 1)
    C = tensor_transpose_tensor_mul(A, B)
    function tensor_transpose_tensor_mul_pullback(C̄)
        C̄ = _seed(T, C̄)
        C̄ isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        NoTangent(), tensor_tensor_transpose_mul(B, C̄), tensor_tensor_mul(A, C̄)
    end
    C, tensor_transpose_tensor_mul_pullback
end
