function ChainRulesCore.rrule(::typeof(vec_tensor_mul), a::AbstractVector{T}, x::AbstractArray{T, 3}) where T 
    b = vec_tensor_mul(a, x)
    function vec_tensor_mul_pullback(b_diff)
        a_diff = @thunk tensor_scalar_product(x, b_diff)
        x_diff = @thunk vec_tensor_mul(a, b_diff)
        NoTangent(), a_diff, x_diff
    end
    b, vec_tensor_mul_pullback
end

# This is computing the sum of scalar products for two tensors
@kernel function tensor_scalar_product_kernel!(a_diff::AbstractVector{T}, x::AbstractArray{T, 3}, b_diff::AbstractArray{T, 3}, range_2, range_3) where T 
    i = @index(Global)
    a_val = zero(T)
    for j = 1:range_2 
        for k = 1:range_3
            a_val += x[i,j,k]*b_diff[i,j,k]
        end
    end
    a_diff[i] = a_val
end

function tensor_scalar_product(x::AbstractArray{T, 3}, b_diff::AbstractArray{T, 3}) where T 
    a_size = size(x, 1)
    backend = KernelAbstractions.get_backend(x)
    a_diff = KernelAbstractions.zeros(backend, T, a_size)
    tensor_scalar_product! = tensor_scalar_product_kernel!(backend)
    tensor_scalar_product!(a_diff, x, b_diff, size(x, 2), size(x, 3), ndrange=size(a_diff))
    a_diff
end

function tensor_scalar_product(x::AbstractArray{T, 3}, b_diff::Thunk) where T
    Thunk(() -> tensor_scalar_product(x, unthunk(b_diff)))
end

vec_tensor_mul(a::AbstractVector, b_diff::Thunk) = Thunk(() -> vec_tensor_mul(a, unthunk(b_diff)))