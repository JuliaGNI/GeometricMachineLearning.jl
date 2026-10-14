using Test
using GeometricMachineLearning
using GeometricMachineLearning: params
using LinearAlgebra: Diagonal, diag, norm
using Zygote: gradient

"""
Assert the shape and the value of the `SymmetricMatrix` leaf of a gradient, taken once with respect
to a `NetworkParameters` wrapper and once with respect to the bare `NamedTuple` it wraps.

Both gradients keep the `SymmetricMatrix` leaf, at any number of accesses to the wrapper. They hold
different quantities. With respect to the bare `NamedTuple`, `Zygote.gradient` projects the dense
gradient ``G`` onto the symmetric matrices, ``(G + G^T)/2``. With respect to the wrapper, the leaf
holds the gradient with respect to the leaf's storage, the lower triangle: its diagonal is that of
``G``, and each off-diagonal entry is ``G_{ij} + G_{ji}``, twice that of the projection.

That is the quantity a training step needs: the loss reads the leaf's storage, and the gradient with
respect to it is what the optimizer steps with, so the wrapper's is the one a pullback returns
(decision 8). The projection is what a gradient of the bare `NamedTuple` gives, which is why a loss
functor takes a `NetworkParameters` and the `Union` arm that also admitted a `NamedTuple` is gone.
"""
function network_parameters_gradient_structure_test(f, ps::NetworkParameters)
    g_wrapped = gradient(_ps -> norm(f(_ps)), ps)[1]
    g_bare = gradient(_ps -> norm(f(_ps)), params(ps))[1]

    a_wrapped = g_wrapped.L1.A
    a_bare = g_bare.L1.A

    @test typeof(a_bare) <: SymmetricMatrix
    @test typeof(a_wrapped) <: SymmetricMatrix

    B = Matrix(a_bare)
    @test Matrix(a_wrapped) ≈ 2B - Diagonal(diag(B))
end
