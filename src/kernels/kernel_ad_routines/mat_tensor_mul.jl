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
   
# get a piece of paper and wirte out these rules! if it's too difficult you can do it with Enzyme
function ChainRulesCore.rrule(::typeof(mat_tensor_mul), B::SkewSymMatrix{T}, A::AbstractArray{T, 3}) where T 
    @assert size(A, 1) == B.n 
    C = mat_tensor_mul(B, A)
    function skew_sym_mul_pullback(C_diff::AbstractArray{T, 3})
        f̄ = NoTangent()
        S_diff = zero(B.S)
        A_diff = zero(C_diff)
        C_diff_copy = copy(C_diff)
        S_copy = copy(B.S)
        A_copy = copy(A)
        C_copy = copy(C)
        Enzyme.autodiff(Enzyme.Reverse, skew_mat_mul!, Enzyme.Const, Enzyme.Duplicated(C_copy, C_diff_copy), Enzyme.Duplicated(S_copy, S_diff), Enzyme.Duplicated(A_copy, A_diff), Enzyme.Const(B.n))

        return f̄, SkewSymMatrix(S_diff, B.n), A_diff 
    end 
    return C, skew_sym_mul_pullback
end

function ChainRulesCore.rrule(::typeof(mat_tensor_mul), B::SymmetricMatrix{T}, A::AbstractArray{T, 3}) where T 
    @assert size(A, 1) == B.n 
    C = mat_tensor_mul(B, A)
    function symmetric_mul_pullback(C_diff::AbstractArray{T, 3})
        f̄ = NoTangent()
        S_diff = zero(B.S)
        A_diff = zero(C_diff)
        C_diff_copy = copy(C_diff)
        S_copy = copy(B.S)
        A_copy = copy(A)
        C_copy = copy(C)
        Enzyme.autodiff(Enzyme.Reverse, symmetric_mat_mul!, Enzyme.Const, Enzyme.Duplicated(C_copy, C_diff_copy), Enzyme.Duplicated(S_copy, S_diff), Enzyme.Duplicated(A_copy, A_diff), Enzyme.Const(B.n))

        return f̄, SymmetricMatrix(S_diff, B.n), A_diff 
    end 
    return C, skew_sym_mul_pullback
end


mat_tensor_mul(B::AbstractMatrix, A::Thunk) = Thunk(() -> mat_tensor_mul(B, unthunk(A)))
    
function tensor_tensor_transpose_mul(B::Thunk, A::AbstractArray{T, 3}) where T 
    Thunk(() -> tensor_tensor_transpose_mul(unthunk(B), A))
end

