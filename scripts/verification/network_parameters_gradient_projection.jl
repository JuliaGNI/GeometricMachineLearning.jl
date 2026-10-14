# Check where a gradient loses the SymmetricMatrix structure of a NetworkParameters leaf.
#
# Run from the repo root:
#   julia --startup-file=no --project=. scripts/verification/network_parameters_gradient_projection.jl
#
# It prints the leaf type of a pullback and of a gradient of two losses that read `ps.L1.A` once and
# twice. With one access both shapes keep the `SymmetricMatrix{Float64, Vector{Float64}}` leaf. With
# two, the pullback of a bare `NamedTuple` drops the leaf to a plain `Matrix{Float64}`, while the
# pullback of a `NetworkParameters` gives a `SymmetricMatrix{Float64, Vector{Float64}}`: the wrapper
# already carries the gradient with respect to the leaf's storage. `Zygote.gradient` projects both
# shapes afterwards, and `ChainRulesCore.ProjectTo` of a `NamedTuple` is a structured projector whose
# leaf maps a `Matrix` back to a `SymmetricMatrix`, while `ProjectTo` of a `NetworkParameters` is a
# projector onto that type, whose leaf stays the `SymmetricMatrix` it wraps.
# CHANGELOG.md records the rule this establishes.

using GeometricMachineLearning
using NeuralNetworkParameters: NetworkParameters
using ChainRulesCore: ProjectTo
using Zygote

S = SymmetricMatrix(rand(3, 3))
nt = (L1 = (A = S,),)
ps = NetworkParameters(nt)

# One getproperty on the argument, the leaf used twice.
one_access = p -> (A = p.L1.A; sum(A) + sum(A))
# Two getproperty calls on the argument.
two_accesses = p -> sum(p.L1.A) + sum(p.L1.A)

leaf(x) = typeof(x.L1.A)
pullback_leaf(f, x) = leaf(Zygote.pullback(f, x)[2](1.0)[1])
gradient_leaf(f, x) = leaf(Zygote.gradient(f, x)[1])

println("pullback, one access,   NamedTuple       => ", pullback_leaf(one_access, nt))
println("pullback, one access,   NetworkParameters => ", pullback_leaf(one_access, ps))
println("pullback, two accesses, NamedTuple       => ", pullback_leaf(two_accesses, nt))
println("pullback, two accesses, NetworkParameters => ", pullback_leaf(two_accesses, ps))
println("gradient, two accesses, NamedTuple       => ", gradient_leaf(two_accesses, nt))
println("gradient, two accesses, NetworkParameters => ", gradient_leaf(two_accesses, ps))
println("ProjectTo(NamedTuple) leaf                => ", typeof(ProjectTo(nt).L1.A))
println("ProjectTo(NamedTuple) applied to a Matrix => ",
    leaf(ProjectTo(nt)((L1 = (A = rand(3, 3),),))))
println("ProjectTo(NetworkParameters)              => ", typeof(ProjectTo(ps)))
