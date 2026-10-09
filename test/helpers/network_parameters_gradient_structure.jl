using Test
using GeometricMachineLearning
using GeometricMachineLearning: params
using LinearAlgebra: norm
using NeuralNetworkParameters: storage_gradient
using Zygote: gradient

"""
Assert the shape and the value of the `SymmetricMatrix` leaf of a gradient taken with respect to a
`NetworkParameters` wrapper.

Since `NeuralNetworkParameters` 0.4 that gradient is the gradient with respect to the leaf's
*storage*: `NeuralNetworkParameters` converts the cotangent of each leaf with `storage_gradient`. A
`SymmetricMatrix` keeps each off-diagonal number once and shows it twice, so its storage gradient
doubles the off-diagonal entries of the cotangent of the dense matrix, and it is a `SymmetricMatrix`
however often the loss reads the leaf. Until 0.4 a second `getproperty` on the wrapper degraded the
leaf to a plain `Matrix` (issue #319).

A `Zygote.gradient` taken with respect to the bare wrapped `NamedTuple` (`params(ps)`) is the
cotangent of the dense matrix, projected back onto a `SymmetricMatrix` by
`ChainRulesCore.ProjectTo`. The two are asserted to differ by exactly `storage_gradient`.
"""
function network_parameters_gradient_structure_test(f, ps::NetworkParameters)
    g_wrapped = gradient(_ps -> norm(f(_ps)), ps)[1]
    g_bare = gradient(_ps -> norm(f(_ps)), params(ps))[1]

    a_wrapped = g_wrapped.L1.A
    a_bare = g_bare.L1.A

    @test typeof(a_bare) <: SymmetricMatrix
    @test typeof(a_wrapped) <: SymmetricMatrix
    @test isapprox(Matrix(a_wrapped), Matrix(storage_gradient(ps.L1.A, a_bare)))
end
