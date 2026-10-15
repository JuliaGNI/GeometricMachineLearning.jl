function ChainRulesCore.rrule(::typeof(tensor_tensor_mul), A::AbstractArray{T, 3},
        B::AbstractArray{T, 3}) where {T}
    @assert axes(A, 2) == axes(B, 1)
    C = tensor_tensor_mul(A, B)
    function tensor_tensor_mul_pullback(C̄)
        C̄ = _seed(T, C̄)
        C̄ isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        NoTangent(), tensor_tensor_transpose_mul(C̄, B), tensor_transpose_tensor_mul(A, C̄)
    end
    C, tensor_tensor_mul_pullback
end
