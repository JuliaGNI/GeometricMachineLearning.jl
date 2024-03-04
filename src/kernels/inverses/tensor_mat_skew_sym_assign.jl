@doc raw"""
A kernel that computes the weighted scalar products of all combinations of vectors in the matrix `Z` except where the two vectors are the same and writes the result into a *tensor of skew symmetric matrices* `C`. 
"""
@kernel function tensor_mat_skew_sym_assign_kernel!(C::AbstractArray{T, 3}, Z::AbstractArray{T, 3}, A::AbstractMatrix{T}) where T
    i, j, k = @index(Global, NTuple)

    temp = zero(T)

    if i > j
        for l in axes(Z, 1)
            for m in axes(Z, 1)
                temp += Z[l, i, k] * A[l, m] * Z[m, j, k]
            end
        end
    end

    C[i, j, k] = temp

    nothing
end

function tensor_mat_skew_sym_assign!(C::AbstractArray{T, 3}, Z::AbstractArray{T, 3}, A::AbstractMatrix{T}) where {T}
    backend = KernelAbstractions.get_backend(Z)

    tensor_mat_skew_sym_assign_k! = tensor_mat_skew_sym_assign_kernel!(backend)

    tensor_mat_skew_sym_assign_k!(C, Z, A, ndrange=size(C))
end

function tensor_mat_skew_sym_assign(Z::AT, A::AbstractMatrix{T}) where {T, AT <: AbstractArray{T, 3}}
    backend = KernelAbstractions.get_backend(Z)

    C = KernelAbstractions.zeros(backend, T, size(Z, 2), size(Z, 2), size(Z, 3))

    tensor_mat_skew_sym_assign!(C, Z, A)

    C
end