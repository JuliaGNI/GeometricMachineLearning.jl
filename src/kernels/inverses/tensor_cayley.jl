function cpu_tensor_cayley(A::AbstractArray{T, 3}) where T 
    one_tensor = init_output(A)
    tensor_tensor_mul(one_tensor - A, cpu_inverse(one_tensor + A))
end

function tensor_cayley2(A::AbstractArray{T, 3}) where T 
    one_tensor = init_output(A)
    tensor_tensor_mul(one_tensor - A, tensor_inverse2(one_tensor + A))
end

function tensor_cayley3(A::AbstractArray{T, 3}) where T 
    one_tensor = init_output(A)
    tensor_tensor_mul(one_tensor - A, tensor_inverse3(one_tensor + A))
end

function tensor_cayley4(A::AbstractArray{T, 3}) where T 
    one_tensor = init_output(A)
    tensor_tensor_mul(one_tensor - A, tensor_inverse4(one_tensor + A))
end

function tensor_cayley5(A::AbstractArray{T, 3}) where T 
    one_tensor = init_output(A)
    tensor_tensor_mul(one_tensor - A, tensor_inverse5(one_tensor + A))
end