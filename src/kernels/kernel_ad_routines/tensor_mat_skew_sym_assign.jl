@kernel function dz_kernel!(dZ::AT, Z::AT, A::AbstractMatrix{T}, dB::AT, sys_dim::Int, seq_length::Int) where {T, AT <: AbstractArray{T, 3}}
    k, n, h = @index(Global, NTuple)

    temp = zero(T)
    for m = 1:sys_dim
        for j = 1:(n - 1)
            temp += A[k, m] * Z[m, j, h] * dB[n, j, h]
        end
        for j = (n + 1):seq_length
            temp += A[m, k] * Z[m, j, h] * dB[j, n, h]
        end
    end

    dZ[k, n, h] = temp

    nothing
end

@kernel function da_kernel!(dA::AT, Z::AT, ::BT, dB::AT, ::Int, seq_length::Int) where {T, AT <: AbstractArray{T, 3}, BT <: AbstractMatrix{T}}
    m, n, h = @index(Global, NTuple)

    temp = zero(T)
    for j = 1:seq_length
        for i = (j+1):seq_length
            temp += Z[m, i, h] * Z[n, j, h] * dB[i, j, h]
        end
    end

    dA[m, n, h] = temp

    nothing
end

function ChainRulesCore.rrule(::typeof(tensor_mat_skew_sym_assign), Z::AbstractArray{T, 3}, A::AbstractArray{T, 2}) where T
    @assert size(A, 1) == size(Z, 1) 
    B = tensor_mat_skew_sym_assign(Z, A)
    function tensor_mat_skew_sym_assign_pullback(dB::AbstractArray{T, 3})
        f̄ = NoTangent()
        backend = KernelAbstractions.get_backend(dB)
        dz! = dz_kernel!(backend)
        da! = da_kernel!(backend)

        dZ = zero(Z)
        dA = KernelAbstractions.zeros(backend, T, size(A)..., size(dB, 3))
        
        sys_dim, seq_length, _ = size(Z)
        
        dz!(dZ, Z, A, dB, sys_dim, seq_length, ndrange = size(dZ))
        da!(dA, Z, A, dB, sys_dim, seq_length, ndrange = size(dA))

        return f̄, dZ, reshape(sum(dA, dims = 3), size(A)) 
    end

    return B, tensor_mat_skew_sym_assign_pullback
end