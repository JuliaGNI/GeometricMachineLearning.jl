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

@doc raw"""
    tensor_mat_skew_sym_assign(Z::AbstractArray{<:Number, 3}, A::AbstractMatrix)

Compute scalar products of columns of ``Z`` along the second axis.

The scalar products are weighted by ``A``.

Scalar products are computed for any two vectors of the form `Z[:, i, k]` and `Z[:, j, k]`, i.e.

```math
    (z^{(i)}, z^{(j)}) \mapsto (z^{(i)})^TAz^{(j)} \text{ for } i > j.
```

The result of this are ``n(n-2)\div2`` scalar products for each index `k` from the third axis. 

These scalar products are written into a lower-triangular matrix and the final output of the function is a tensor of these lower-triangular matrices. 

This is used in [`VolumePreservingAttention`](@ref) when `skew_sym` is set to `false`.
"""
function tensor_mat_skew_sym_assign(Z::AT, A::AbstractMatrix{T})::AT where {T, AT <: AbstractArray{T, 3}}
    backend = KernelAbstractions.get_backend(Z)

    C = KernelAbstractions.zeros(backend, T, size(Z, 2), size(Z, 2), size(Z, 3))

    tensor_mat_skew_sym_assign!(C, Z, A)

    C
end