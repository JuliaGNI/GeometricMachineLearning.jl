import Random, Test, Lux

using GeometricMachineLearning, Test

model = MultiHeadAttention(64, 8, Stiefel=true)
ps, st = Lux.setup(Random.default_rng(), model)
tol = eps(Float32)*10

check_type(ps::NamedTuple) = apply_toNT(ps, check_type)
check_type(A::AbstractMatrix) = @test typeof(A) <: StiefelManifold

check_type(ps)

GeometricMachineLearning.check(ps::NamedTuple) = apply_toNT(ps, check)

check_tol(a::Real) = @test a < tol
check_tol(ps::NamedTuple) = apply_toNT(ps, check_tol)

check_tol(check(ps))

######## check if the gradients are set up the correct way 
psᵧ, stᵧ = Lux.setup(TrivialInitRNG(), model)
check_type_grad(ps::NamedTuple) = apply_toNT(ps, check_type_grad)
check_type_grad(B::AbstractMatrix) = @test typeof(B) <: StiefelLieAlgHorMatrix

check_type_grad(psᵧ)