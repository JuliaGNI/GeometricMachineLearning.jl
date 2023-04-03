"""
This file contains various gradient that are not related to any explicit parametrization.
An example would be the computation of gradᵍL ∈ TᵤSp(2n,2N) ⊂ ℝ²ᴺˣ²ⁿ for arbitrary U (Tₑ is implemented as a separate vector space!)
"""
#this function returns a NamedTuple containing the weight of the layers as is used for the parameters in Lux.
function gradᵍ(U::AbstractMatrix, e_grad::AbstractMatrix, J::AbstractMatrix)
    (weight = e_grad * U' * U + J * U * e_grad' * J * U,)
end


function gradʰ(U::AbstractMatrix, e_grad::AbstractMatrix, J::AbstractMatrix)

end