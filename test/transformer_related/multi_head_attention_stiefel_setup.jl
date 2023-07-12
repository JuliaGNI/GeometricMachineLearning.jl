import Random, Test, Lux, LinearAlgebra

using GeometricMachineLearning, Test

model = MultiHeadAttention(64, 8, Stiefel=true)
ps, st = Lux.setup(Random.default_rng(), model)
tol = 10*eps(Float32)

function check_setup(A::AbstractMatrix)
    @test typeof(A) <: StiefelManifold
    @test check(A) < tol
end

check_setup(ps::NamedTuple) = apply_toNT(ps, check_setup)

check_setup(ps)

######## check if the gradients are set up the correct way 
function check_grad_setup(B::AbstractMatrix)
    @test typeof(B) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(B) < tol
end
check_grad_setup(ps::NamedTuple) = apply_toNT(ps, check_grad_setup)

psᵧ, stᵧ = Lux.setup(TrivialInitRNG(), model)
check_grad_setup(psᵧ)
