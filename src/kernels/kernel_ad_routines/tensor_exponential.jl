"""
This implements the custom pullbacks relevant for tensor_exponential; not easy.
"""

#=
# the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(tensor_exponential), A::AbstractArray{T, 3}) where T
    output = tensor_exponential(A)
    m, m2, batch_size = size(A)
    function tensor_exponential_pullback(B_diff)
        f̄ = NoTangent()
        A_diff = @thunk tensor_exponential_differential(B_diff)
        return f̄, A_diff
    end
    return B, tensor_exponential_pullback
end

function tensor_exponential_differential(B_diff::AbstractArray{T, 3}) where T 
    ...
end
=#

function ChainRulesCore.rrule(::typeof(allocate_matrix), A::AbstractArray{T, 3}, k) where T 
    B = allocate_matrix(A, k)
    backend = KernelAbstractions.get_backend(A)
    function allocate_matrix_pullback(B_diff)
        f = NoTangent()
        A_diff = KernelAbstractions.zeros(backend, T, size(A)...)
        allocate_tensor!(A_diff, B_diff, k)
        return f, A_diff, NoTangent()
    end
    return B, allocate_matrix_pullback
end

function ChainRulesCore.rrule(::typeof(allocate_tensor), B::AbstractMatrix{T}, batch_size, k) where T
    A = allocate_tensor(B, batch_size, k)
    backend = KernelAbstractions.get_backend(B)
    function allocate_tensor_pullback(A_diff)
        f = NoTangent()
        B_diff = @thunk allocate_matrix(A_diff, k)
        return f, B_diff, NoTangent(), NoTangent()
    end
    return A, allocate_tensor_pullback
end

function allocate_tensor(A::Thunk, batch_size, k) 
    Thunk(() -> allocate_tensor(unthunk(A), batch_size, k))
end

function allocate_matrix(A::Thunk, k)
    Thunk(()->allocate_matrix(unthunk(A), k))
end