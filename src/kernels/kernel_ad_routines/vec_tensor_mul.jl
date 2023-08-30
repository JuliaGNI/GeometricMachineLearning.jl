function ChainRulesCore.rrule(::typeof(vec_tensor_mul), a::AbstractVector{T}, x::AbstractArray{T, 3}) where T 
    b = vec_tensor_mul(a, x)
    function vec_tensor_mul_pullback(b_diff)
        a_diff = @thunk tensor_scalar_product(x, b_diff)
        x_diff = @thunk vec_tensor_mul(a, b_diff)
        NoTangent(), a_diff, x_diff
    end
    b, vec_tensor_mul_pullback
end

@kernel function tensor_scalar_product_kernel!(a_diff::AbstractVector{T}, x::AbstractArray{T, 3}, b_diff::AbstractArray{T, 3}) where T 
    i, j, k = @index(Global, NTuple)
    a[i] += x[i,j,k]*b_diff[i,j,k]
end

function tensor_scalar_product(x::AbstractArrau{T, 3}, b_diff::AbstractArray{T, 3}) where T 
    a_size = size(x, 1)
    backend = KernelAbstractions.get_backend(x)
    a_diff = KernelAbstractions.zeros(backend, T, a_size)
    tensor_scalar_product! = tensor_scalar_product_kernel!(backend)
    tensor_scalar_product!(a_diff, x, b_diff)
    a_diff
end