"""
This implements the custom pullback tor mat_tensor_mul
"""

#the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(mat_tensor_mul), B::AbstractMatrix{T}, A::AbstractArray{T, 3}) where T
    @assert axes(A, 1) == axes(B, 2)
    C = mat_tensor_mul(B, A)
    function mat_tensor_mul_pullback(C_diff)
        f̄ = NoTangent()
        #tensor_transpose_mat_mul
        B_diff = @thunk sum(tensor_tensor_transpose_mul(C_diff, A), dims=3)
        A_diff = @thunk mat_tensor_mul(B', C_diff)
        return f̄, B_diff, A_diff
    end
    return C, mat_tensor_mul_pullback
end

################################ lower triangular 

@kernel function lower_da_kernel!(dA::AT, S::AbstractVector{T}, dC::AT) where {T, AT <: AbstractArray{T, 3}} 
    l, m, h = @index(Global, NTuple)

    temp = zero(T)
    for i = (l+1):size(dA, 1)
        temp += S[(i-2) * (i-1) ÷ 2 + l] * dC[i, m, h]
    end

    dA[l, m, h] = temp 

    nothing
end

@kernel function lower_ds_kernel!(dS::AbstractMatrix{T}, A::AT, dC::AT) where {T, AT <: AbstractArray{T, 3}}
    l, h = @index(Global, NTuple)
    n = size(A, 1)

    temp = zero(T)
    for j in axes(A, 2)
        for i in 2:n 
            i_sum = (i - 2) * (i - 1) ÷ 2
            temp = (i_sum < l < (i_sum + i)) ? temp + A[l - i_sum, j, h] * dC[i, j, h] : temp
        end
    end
    

    dS[l, h] = temp 

    nothing
end

function ChainRulesCore.rrule(::typeof(lo_mat_mul), S::AbstractVector{T}, A::AbstractArray{T, 3}, n::Int) where T 
    C = lo_mat_mul(S, A, n)
    function lo_mat_mul_pullback(dC::AbstractArray{T, 3}) 
        f̄ = NoTangent()
        backend = KernelAbstractions.get_backend(dC)
        lower_da! = lower_da_kernel!(backend)
        lower_ds! = lower_ds_kernel!(backend)

        dA = zero(A)
        dS = KernelAbstractions.zeros(backend, T, length(S), size(dC, 3))

        lower_da!(dA, S, dC, ndrange=size(dA))
        lower_ds!(dS, A, dC, ndrange=size(dS))

        return f̄, reshape(sum(dS, dims = 2), length(S)), dA, NoTangent()
    end

    return C, lo_mat_mul_pullback
end

function ChainRulesCore.rrule(::typeof(mat_tensor_mul), B::LowerTriangular{T}, A::AbstractArray{T, 3}) where T
    @assert size(A, 1) == B.n 
    C = mat_tensor_mul(B, A)
    function lower_triangular_mul_pullback(dC::AbstractArray{T, 3})
        f̄, dS, dA, _  = rrule(lo_mat_mul, B.S, A, B.n)[2](dC)

        return f̄, LowerTriangular(dS, B.n), dA 
    end 
    return C, lower_triangular_mul_pullback
end

################################ upper triangular 

@kernel function upper_da_kernel!(dA::AT, S::AbstractVector{T}, dC::AT) where {T, AT <: AbstractArray{T, 3}} 
    l, m, h = @index(Global, NTuple)

    temp = zero(T)
    for i = 1:(l-1)
        temp += S[(l - 2) * (l - 1) ÷ 2 + i] * dC[i, m, h]
    end

    dA[l, m, h] = temp 

    nothing
end

@kernel function upper_ds_kernel!(dS::AbstractMatrix{T}, A::AT, dC::AT) where {T, AT <: AbstractArray{T, 3}}
    l, h = @index(Global, NTuple)
    n = size(A, 1)

    temp = zero(T)
    for k in 1:n
        k_sum = (k - 2) * (k - 1) ÷ 2
        if k_sum < l < (k_sum + k)
            for j in axes(A, 2) 
                temp += A[k, j, h] * dC[l - k_sum, j, h] 
            end
        end
    end
    

    dS[l, h] = temp 

    nothing
end

function ChainRulesCore.rrule(::typeof(up_mat_mul), S::AbstractVector{T}, A::AbstractArray{T, 3}, n::Int) where T 
    C = up_mat_mul(S, A, n)
    function up_mat_mul_pullback(dC::AbstractArray{T, 3}) 
        f̄ = NoTangent()
        backend = KernelAbstractions.get_backend(dC)
        upper_da! = upper_da_kernel!(backend)
        upper_ds! = upper_ds_kernel!(backend)

        dA = zero(A)
        dS = KernelAbstractions.zeros(backend, T, length(S), size(dC, 3))

        upper_da!(dA, S, dC, ndrange=size(dA))
        upper_ds!(dS, A, dC, ndrange=size(dS))

        return f̄, reshape(sum(dS, dims = 2), length(S)), dA, NoTangent()
    end

    return C, up_mat_mul_pullback
end

function ChainRulesCore.rrule(::typeof(mat_tensor_mul), B::UpperTriangular{T}, A::AbstractArray{T, 3}) where T
    @assert size(A, 1) == B.n 
    C = mat_tensor_mul(B, A)
    function upper_triangular_mul_pullback(dC::AbstractArray{T, 3})
        f̄, dS, dA, _ = rrule(up_mat_mul, B.S, A, B.n)[2](dC)

        return f̄, UpperTriangular(dS, B.n), dA 
    end 
    return C, upper_triangular_mul_pullback
end

################################ skew-symmetric 

function ChainRulesCore.rrule(::typeof(skew_mat_mul), S::AbstractVector{T}, A::AbstractArray{T, 3}, n::Int) where T 
    C = skew_mat_mul(S, A, n)
    function skew_mat_mul_pullback(dC::AbstractArray{T, 3})
        f̄ = NoTangent()
        backend = KernelAbstractions.get_backend(dC)
        lower_da! = lower_da_kernel!(backend)
        lower_ds! = lower_ds_kernel!(backend)
        upper_da! = upper_da_kernel!(backend)
        upper_ds! = upper_ds_kernel!(backend)

        dA_lower = zero(A)
        dS_lower = KernelAbstractions.zeros(backend, T, length(S), size(dC, 3))
        dA_upper = zero(A)
        dS_upper = KernelAbstractions.zeros(backend, T, length(S), size(dC, 3))

        lower_da!(dA_lower, S, dC, ndrange=size(dA_lower))
        lower_ds!(dS_lower, A, dC, ndrange=size(dS_lower))
        upper_da!(dA_upper, S, dC, ndrange=size(dA_upper))
        upper_ds!(dS_upper, A, dC, ndrange=size(dS_upper))

        return f̄, reshape(sum(dS_lower - dS_upper, dims = 2), length(S)), dA_lower - dA_upper, NoTangent()
    end

    return C, skew_mat_mul_pullback
end

function ChainRulesCore.rrule(::typeof(mat_tensor_mul), B::SkewSymMatrix{T}, A::AbstractArray{T, 3}) where T 
    @assert size(A, 1) == B.n 
    C = mat_tensor_mul(B, A)
    function skew_sym_mul_pullback(dC::AbstractArray{T, 3})
        f̄, dS, dA, _ = rrule(skew_mat_mul, B.S, A, B.n)[2](dC)

        return f̄, SkewSymMatrix(dS, B.n), dA 
    end 
    return C, skew_sym_mul_pullback
end

################################ symmetric 

@kernel function symmetric_da_kernel!(dA::AT, S::AbstractVector{T}, dC::AT) where {T, AT <: AbstractArray{T, 3}}
    l, m, h = @index(Global, NTuple)
    temp = zero(T)
    for i = l:size(dA, 1)
        i_sum = (i - 1) * i ÷ 2
        temp += S[i_sum + l] * dC[i, m, h]
    end
    l_sum = (l - 1) * l ÷ 2
    for i = 1:(l - 1)
        temp += S[l_sum + i] * dC[i, m, h]
    end 
    dA[l, m, h] = temp

    nothing
end

@kernel function symmetric_ds_kernel!(dS::AbstractMatrix{T}, A::AT, dC::AT) where {T, AT <: AbstractArray{T, 3}}
    l, h = @index(Global, NTuple)
    temp = zero(T)
    for i in axes(dC, 1)
        sum_i = (i - 1) * i ÷ 2
        if sum_i < l
            for j in axes(dC, 1)
                if l < (sum_i + i) 
                    temp += A[l - sum_i, j, h] * dC[i, j, h]
                    temp += A[i, j, h] * dC[l - sum_i, j, h]
                end
                if l == (sum_i + i)
                    temp += A[l - sum_i, j, h] * dC[i, j, h]
                end
            end
        end 
    end
    dS[l, h] = temp 

    nothing 
end

function ChainRulesCore.rrule(::typeof(symmetric_mat_mul), S::AbstractVector{T}, A::AbstractArray{T, 3}, n::Int) where  T 
    C = symmetric_mat_mul(S, A, n)
    function symmetric_mat_mul_pullback(dC::AbstractArray{T, 3}) 
        backend = KernelAbstractions.get_backend(dC)
        symmetric_da! = symmetric_da_kernel!(backend)
        symmetric_ds! = symmetric_ds_kernel!(backend)

        dA = zero(A)
        dS = KernelAbstractions.zeros(backend, T, length(S), size(dC, 3))

        symmetric_da!(dA, S, dC, ndrange = size(dA))
        symmetric_ds!(dS, A, dC, ndrange = size(dS))

        NoTangent(), reshape(sum(dS, dims = 2), length(S)), dA, NoTangent()
    end

    C, symmetric_mat_mul_pullback 
end

function ChainRulesCore.rrule(::typeof(mat_tensor_mul), B::SymmetricMatrix{T}, A::AbstractArray{T, 3}) where T 
    @assert size(A, 1) == B.n 
    C = mat_tensor_mul(B, A)
    function symmetric_mul_pullback(dC::AbstractArray{T, 3})
        f̄, dS, dA, _ = rrule(symmetric_mat_mul, B.S, A, B.n)[2](dC)        

        return f̄, SymmetricMatrix(dS, B.n), dA 
    end 

    return C, symmetric_mul_pullback
end

############### Thunks

mat_tensor_mul(B::AbstractMatrix, A::Thunk) = Thunk(() -> mat_tensor_mul(B, unthunk(A)))
    
function tensor_tensor_transpose_mul(B::Thunk, A::AbstractArray{T, 3}) where T 
    Thunk(() -> tensor_tensor_transpose_mul(unthunk(B), A))
end

