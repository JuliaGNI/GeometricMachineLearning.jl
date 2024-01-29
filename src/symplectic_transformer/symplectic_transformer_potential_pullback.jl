@doc raw"""
The greadient of `symplectic_transformer_potential_gradient` is done with Enzyme, and not with Zygote as this allows differentiating kernels from KernelAbstractions. 
"""
function ChainRulesCore.rrule(::typeof(symplectic_transformer_potential_gradient), Z::AbstractArray, A::AbstractMatrix)
    output = symplectic_transformer_potential_gradient(Z, A)
    function symplectic_transformer_potential_gradient!(output, Z, A)
        output .= symplectic_transformer_potential_gradient(Z, A)
        nothing 
    end
    function symplectic_transformer_potential_gradient_pullback(output_diff)
        Z_diff = zero(Z)
        A_diff = zero(A)
        Enzyme.autodiff(Enzyme.Reverse, symplectic_transformer_potential_gradient!, Enzyme.Const, Enzyme.Duplicated(output, output_diff), Enzyme.Duplicated(Z, Z_diff), Enzyme.Duplicated(A, A_diff))
        NoTangent(), Z_diff, A_diff
    end
    output, symplectic_transformer_potential_gradient_pullback
end