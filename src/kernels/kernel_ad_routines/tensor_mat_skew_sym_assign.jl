@kernel function dz_kernel!(dZ::AT, Z::AT, A::AbstractMatrix{T}, dB::AT, sys_dim::Int, seq_length::Int) where {T, AT <: AbstractArray{T, 3}}
    k, l, h = @index(Global, NTuple)

    temp = zero(T)
    for n = 1:sys_dim 
        for j = 1:(l-1)
            temp += A[k, n] * Z[n, j, h] * dB[l, j, h]
        end
        for i = (l+1):seq_length
            temp += A[n, k] * Z[n, i, h] * dB[i, l, h]
        end
    end

    dZ[k, l, h] = temp

    nothing
end

@kernel function da_kernel!(dA::BT, Z::AT, ::BT, dB::AT, ::Int, seq_length::Int) where {T, AT <: AbstractArray{T, 3}, BT <: AbstractMatrix{T}}
    m, n = @index(Global, NTuple)

    temp = zero(T)
    for h in axes(Z, 3)
        for j = 1:seq_length
            for i = (j+1):seq_length
                temp += Z[m, i, h] * Z[n, j, h] * dB[i, j, h]
            end
        end
    end

    dA[m, n] = temp

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
        dA = zero(A)
        
        sys_dim, seq_length, _ = size(dB)
        
        dz!(dZ, Z, A, dB, sys_dim, seq_length, ndrange = size(dZ))
        da!(dA, Z, A, dB, sys_dim, seq_length, ndrange = size(dA))

        return f̄, dZ, dA 
    end

    return B, tensor_mat_skew_sym_assign_pullback
end