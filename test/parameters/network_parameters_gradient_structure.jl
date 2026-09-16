using Test
using GeometricMachineLearning
using GeometricMachineLearning: params
using LinearAlgebra: norm
using Zygote: gradient

"""
Assert which shape the `SymmetricMatrix` leaf of a gradient has, and that the shape is cosmetic.

A gradient taken with respect to a `NetworkParameters` wrapper keeps its `SymmetricMatrix` leaf
only when the loss expression performs a single `getproperty` on that wrapper. A second access
degrades the leaf to a plain `Matrix`. What decides the split is the number of `getproperty` calls
on the wrapper itself -- not how often the leaf is used, and not how often the plain `NamedTuple`s
below the wrapper are accessed. `p -> (A = p.L1.A; sum(A) + sum(A))` uses the leaf twice through one
access and keeps the structure; `p -> sum(p.L1.A) + sum(p.L1.A)` loses it. A `Zygote.gradient` taken
with respect to the bare wrapped `NamedTuple` (`params(ps)`) keeps the structure at any access
count, because `gradient` projects its result and `ChainRulesCore.ProjectTo` restores the leaf for a
`NamedTuple` but is `identity` for a `NetworkParameters`. The underlying `Zygote.pullback` loses the
structure on the second access either way. `CHANGELOG.md` records the experiment.

A wrapper unwrapped with `values` rather than `getproperty` -- which is how `Chain` hands a layer
its parameters -- counts differently: there the accesses on the plain `NamedTuple` below decide.
See the layer testset in
`test/parameters/symplectic_attention_network_parameters_gradient.jl`.

`preserves_structure` states which side of that split `f` sits on. Both gradients are asserted to
agree numerically either way, so losing the structure is a change of type and not of value.
"""
function network_parameters_gradient_structure_test(
        f, ps::NetworkParameters, preserves_structure::Bool)
    g_wrapped = gradient(_ps -> norm(f(_ps)), ps)[1]
    g_bare = gradient(_ps -> norm(f(_ps)), params(ps))[1]

    a_wrapped = g_wrapped.L1.A
    a_bare = g_bare.L1.A

    @test typeof(a_bare) <: SymmetricMatrix
    if preserves_structure
        @test typeof(a_wrapped) <: SymmetricMatrix
    else
        @test_broken typeof(a_wrapped) <: SymmetricMatrix
    end

    @test isapprox(Matrix(a_wrapped), Matrix(a_bare))
end
