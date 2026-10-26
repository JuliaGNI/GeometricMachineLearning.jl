# The canonical Poisson tensor as the dense matrix it is: `[0 I(n); -I(n) 0]` of `n2 × n2`, with
# `n = n2 ÷ 2`, the `q` block first. The symplecticity tests compare a decoder Jacobian against it,
# which needs nothing but that matrix, so the tests build it here rather than depend on a type of
# the package for four blocks.

using LinearAlgebra: I

function poisson_matrix(n2::Integer, ::Type{T} = Float64) where {T}
    iseven(n2) || throw(ArgumentError("the dimension of a Poisson tensor is even, got $n2"))
    n = n2 ÷ 2
    [zeros(T, n, n) Matrix{T}(I, n, n); -Matrix{T}(I, n, n) zeros(T, n, n)]
end
