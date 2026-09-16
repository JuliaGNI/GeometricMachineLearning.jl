# Check where a gradient loses the SymmetricMatrix structure of a NetworkParameters leaf.
#
# Run from the repo root:
#   julia --startup-file=no --project=. scripts/verification/network_parameters_gradient_projection.jl
#
# The reverse pass is not what distinguishes the wrapper from the bare NamedTuple: Zygote.pullback
# drops the leaf to a plain Matrix on the second getproperty for both. Zygote.gradient projects its
# result afterwards, and ChainRulesCore.ProjectTo of a NamedTuple is a structured projector whose
# leaf maps the Matrix back to a SymmetricMatrix, while ProjectTo of a NetworkParameters falls back
# to identity. CHANGELOG.md records the rule this establishes.

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
