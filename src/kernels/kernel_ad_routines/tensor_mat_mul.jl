"""
This implements the custom pullback for tensor_mat_mul
"""

#the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(tensor_mat_mul), A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where T
    @assert axes(A, 2) == axes(B, 1)
    C = tensor_mat_mul(A, B)
    function tensor_mat_mul_pullback(C_diff)
        f̄ = NoTangent()
        #tensor_transpose_mat_mul
        A_diff = @thunk tensor_mat_mul(C_diff, B')
        B_diff = @thunk sum(tensor_transpose_tensor_mul(A, C_diff), dims=3)
        return f̄, A_diff, B_diff
    end
    return C, tensor_mat_mul_pullback
end

tensor_mat_mul(A::Thunk, B::AbstractMatrix) = Thunk(() -> tensor_mat_mul(unthunk(A), B))

function tensor_transpose_tensor_mul(A::AbstractArray{T, 3}, B::Thunk) where T 
    Thunk(() -> tensor_transpose_tensor_mul(A, unthunk(B)))
end

############### Symmetric (right mul)

@kernel function symmetric_right_da_kernel!(dA::AT, S::AbstractVector{T}, dC::AT) where {T, AT <: AbstractArray{T, 3}}
    l, m, h = @index(Global, NTuple)
    
    temp = zero(T)

    for j = 1:m
        temp += S[(m + 1) * m ÷ 2 + j] * dC[l, j, h]
    end
    for j = (m+1):size(dA, 2)
        temp += S[(j - 1) * j ÷ 2 + m] * dC[l, j, h]
    end

    dA[l, m, h] = temp

    nothing
end

@kernel function symmetric_right_ds_kernel!(dS::AbstractMatrix{T}, B::AT, dC::AT) where {T, AT <: AbstractArray{T, 3}}
    l, h = @index(Global, NTuple)
    temp = zero(T)
    
    for i in axes(dC, 1)
        for k in axes(dC, 2)
            sum_k = (k - 1) * k ÷ 2
            temp += l ≤ k + sum_k ? B[i, k, h] * dC[i, l - sum_k, h] : zero(T)
        end
        for j in axes(dC, 2)
            sum_j = (j - 1) * j ÷ 2
            temp += l < sum_j + j ? B[i, l - sum_j, h] * dC[i, j, h] : zero(T)
        end
    end

    dS[l, h] = temp 

    nothing 
end

function ChainRulesCore.rrule(::typeof(symmetric_mat_right_mul), A::AbstractArray{T, 3}, S::AbstractVector{T}, n::Int) where  T 
    C = symmetric_mat_right_mul(A, S, n)
    function symmetric_mat_mul_pullback(dC::AbstractArray{T, 3}) 
        backend = KernelAbstractions.get_backend(dC)
        symmetric_right_da! = symmetric_right_da_kernel!(backend)
        symmetric_right_ds! = symmetric_right_ds_kernel!(backend)

        dA = zero(A)
        dS = KernelAbstractions.zeros(backend, T, length(S), size(dC, 3))

        symmetric_right_da!(dA, S, dC, ndrange = size(dA))
        symmetric_right_ds!(dS, A, dC, ndrange = size(dS))

        NoTangent(), reshape(sum(dS, dims = 2), length(S)), dA, NoTangent()
    end

    C, symmetric_mat_mul_pullback 
end

function ChainRulesCore.rrule(::typeof(tensor_mat_mul), A::AbstractArray{T, 3}, B::SymmetricMatrix{T}) where T 
    @assert size(A, 2) == B.n 
    C = tensor_mat_mul(A, B)
    function symmetric_right_mul_pullback(dC::AbstractArray{T, 3})
        f̄, dA, dS, _ = rrule(symmetric_mat_right_mul, A, B.S, B.n)[2](dC)        

        return f̄, dA, SymmetricMatrix(dS, B.n) 
    end 

    return C, symmetric_mul_pullback
end