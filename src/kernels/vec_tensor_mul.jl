@kernel function vec_tensor_mul_kernel!(b::AbstractArray{T, 3}, a::AbstractVector{T}, x::AbstractArray{T, 3}) where T 
    i, j, k = @index(Global, NTuple)
    b[i,j,k] = x[i,j,k]*a[i]
end

function vec_tensor_mul(a::AbstractVector{T}, x::AbstractArray{T, 3}) where T 
    b = similar(x)
    backend = KernelAbstractions.get_backend(x)
    vec_tensor_mul! = vec_tensor_mul_kernel!(backend)
    vec_tensor_mul!(b, a, x, ndrange=size(x))
    b 
end 