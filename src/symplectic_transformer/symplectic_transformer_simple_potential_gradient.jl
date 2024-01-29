function symplectic_transformer_simple_potential_gradient(Z::AbstractArray{T, 3}, A::AbstractMatrix{T}) where T
    A_sym = T(.5) * (A + matrix_transpose(A))
    AZ = mat_tensor_mul(A_sym, Z)
    tensor_tensor_mul(Z, tensor_transpose_tensor_mul(Z, AZ)) + 
    tensor_tensor_mul(AZ, tensor_transpose_tensor_mul(Z, Z))
end