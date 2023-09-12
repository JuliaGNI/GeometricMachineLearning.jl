"""
For now this implements the inverse for a batch in serial (fix!!!)
"""

using ChainRulesCore, KernelAbstractions

@kernel function assign_matrix_kernel!(B::AbstractMatrix{T}, A::AbstractArray{T, 3}, k::Integer) where T 
    i, j = @index(Global, NTuple)
    B[i,j] = A[i,j,k]
end
@kernel function assign_tensor_kernel!(A::AbstractArray{T, 3}, B::AbstractMatrix{T}, k::Integer) where T 
    i, j = @index(Global, NTuple)
    A[i,j,k] = B[i,j]
end
function assign_matrix(A::AbstractArray{T, 3}, k::Integer) where T 
    m1, m2, m3 = size(A)
    backend = KernelAbstractions.get_backend(A)
    B = allocate(backend, T, m1, m2)
    assign_matrix! = assign_matrix_kernel!(backend)
    assign_matrix!(B, A, k, ndrange=size(B))
    B
end
function assign_tensor!(A::AbstractArray{T,3}, B::AbstractMatrix{T}, total_length::Integer, k::Integer) where T 
    m1, m2, _ = size(A)
    backend = KernelAbstractions.get_backend(B)
    assign_tensor_apply! = assign_tensor_kernel!(backend)
    assign_tensor_apply!(A, B, k, ndrange=size(B))
end
function assign_tensor!(A::AbstractArray{T,3}, B::AbstractMatrix{T}, k::Integer) where T 
    total_length = size(A, 3)
    assign_tensor!(A, B, total_length, k)
end
function assign_tensor(B::AbstractMatrix{T}, total_length::Integer, k::Integer) where T 
    backend = KernelAbstractions.get_backend(B)
    m1, m2 = size(B)
    A = KernelAbstractions.zeros(backend, T, m1, m2, total_length)
    assign_tensor!(A, B, total_length, k)
    A
end

function ChainRulesCore.rrule(::typeof(assign_matrix), A::AbstractArray{T, 3}, k) where T 
    B = assign_matrix(A, k)
    function assign_matrix_pullback(B_diff)
        total_length = size(A, 3)
        A_diff = assign_tensor(B_diff, total_length, k)
        NoTangent(), A_diff, NoTangent()
    end
    B, assign_matrix_pullback
end

function ChainRulesCore.rrule(::typeof(assign_tensor), B::AbstractMatrix{T}, total_length::Integer, k::Integer) where  T 
    A = assign_tensor(B, total_length, k)
    function assign_tensor_pullback(A_diff)
        B_diff = @thunk assign_matrix(A_diff, k)
        return NoTangent(), B_diff, NoTangent(), NoTangent()
    end
    A, assign_tensor_pullback 
end

function tensor_inverse(A::AbstractArray{T, 3}) where T 
    A_inv = zero(A)
    total_length = size(A, 3)
    #for k in 1:total_length 
    #    B = assign_matrix(A, k)
    #    B_inv = B\one(B)
    #    A_inv += assign_tensor(B_inv, total_length, k)
    #end

    for k in axes(A_inv, 3)
        B = @view A[:,:,k]
        A_inv[:,:,k] .= B \ one(B) 
    end

    A_inv
end 

assign_matrix(A::Thunk, k) = Thunk(() -> assign_matrix(unthunk(A), k))
assign_tensor(B::Thunk, total_length, k) = Thunk(() -> assign_tensor(unthunk(B), total_length, k))
function assign_tensor!(A::AbstractArray{T, 3}, B::Thunk, total_length, k) where T 
    Thunk(() -> assign_tensor!(A, unthunk(B), total_length, k))
end

function tensor_cayley(A::AbstractArray{T, 3}) where T 
    one_tensor = init_output(A)
    tensor_tensor_mul(one_tensor - A, tensor_inverse(one_tensor + A))
end