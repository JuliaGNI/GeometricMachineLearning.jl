"""
This function tests if the lift really maps to the invariant horizontal component of the Lie algebra.
"""

using LinearAlgebra
using Test

include("../src/arrays/skew_sym.jl")
include("../src/optimizers/householder.jl")
include("../src/optimizers/manifold_types.jl")
include("../src/arrays/stiefel_lie_alg_hor.jl")
include("../src/optimizers/lie_alg_lifts.jl")
include("../src/arrays/auxiliary.jl")

function stiefel_lift_test(N, n, ε=1e-12)
    Y = StiefelManifold(N, n)
    V = SkewSymMatrix(randn(N,N))*Y
    #global_rep gives two outputs: a householder elment and the lifted element of the Lie algebra 
    V_lift = global_rep_test(Y,V)[2]
    @test norm(V_lift - StiefelLieAlgHorMatrix(SkewSymMatrix(V_lift), n))/N < ε
end 