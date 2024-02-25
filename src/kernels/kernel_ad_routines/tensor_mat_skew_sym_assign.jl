function ChainRulesCore.rrule(::typeof(tensor_mat_skew_sym_assign), Z::AbstractArray{T, 3}, A::AbstractArray{T, 2}) where T
    @assert size(A, 1) == size(Z, 1) 
    C = tensor_mat_skew_sym_assign(Z, A)
    function tensor_mat_skew_sym_assign_pullback(C_diff::AbstractArray{T, 3})
        f̄ = NoTangent()
        A_diff = zero(A)
        Z_diff = zero(Z)
        C_diff_copy = copy(C_diff)
        A_copy = copy(A)
        Z_copy = copy(Z)
        C_copy = copy(C)
        Enzyme.autodiff(Enzyme.Reverse, tensor_mat_skew_sym_assign!, Enzyme.Const, Enzyme.Duplicated(C_copy, C_diff_copy), Enzyme.Duplicated(Z_copy, Z_diff), Enzyme.Duplicated(A_copy, A_diff))

        return f̄, Z_diff, A_diff 
    end 
    return C, tensor_mat_skew_sym_assign_pullback
end