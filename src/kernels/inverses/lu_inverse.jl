using KernelAbstractions, CUDA, LinearAlgebra
import Random

@kernel function lu_inverse_kernel!(B, A) 
    k = @index(Global)
    A_temp = A[:, :, k]
    B[:, :, k] .= CUDA.inv(A_temp)

    nothing
end

function lu_inverse(A::AbstractArray)
    B = zero(A)
    backend = KernelAbstractions.get_backend(A)

    lu_inverse! = lu_inverse_kernel!(backend)
    lu_inverse!(B, A, ndrange=size(A, 3))

    B 
end

@kernel function matrix_multiplication_kernel!(B::AbstractArray{T}, A::AbstractArray{T}) where T  
    k = @index(Global)
    @views A_temp = A[:, :, k]
    @views B_temp = B[:, :, k]
    mul!(B_temp, A_temp, A_temp) 

    nothing
end

function matrix_multiplication(A::AbstractArray)
    B = zero(A)
    backend = KernelAbstractions.get_backend(A)

    matrix_multiplication! = matrix_multiplication_kernel!(backend)
    matrix_multiplication!(B, A, ndrange=size(A, 3))

    B 
end

A = CUDA.rand(4, 4, 3)
matrix_multiplication(A)