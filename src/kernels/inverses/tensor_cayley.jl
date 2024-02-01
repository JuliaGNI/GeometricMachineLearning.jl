function tensor_cayley4(A::AbstractArray{T, 3}) where T 
    one_tensor = init_output(A)
    tensor_tensor_mul(one_tensor - A, tensor_inverse4(one_tensor + A))
end

function tensor_cayley6(A::AbstractArray{T, 3}) where T 
    one_tensor = init_output(A)
    tensor_tensor_mul(one_tensor - A, tensor_inverse6(one_tensor + A))
end