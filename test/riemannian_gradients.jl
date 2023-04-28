"""
This implements tests for the Riemannian gradients.

TODO: find correct expression for Riemannian gradient for the canonical metric!
"""

using Zygote
using Test

include("../src/optimizers/auxiliary_gradients.jl")
include("../src/optimizers/manifold_types.jl")



function stiefel_riemannian_gradient_test(N::Int, n::Int)

end