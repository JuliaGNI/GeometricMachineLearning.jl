import Random, Test, Lux

using GeometricMachineLearning, Test

model = MultiHeadAttention(64, 8, Stiefel=true)
psᵧ, stᵧ = Lux.setup(TrivialInitRNG(), model)

ps₂ = retraction(model, psᵧ)

check_type(ps₂::NamedTuple) = apply_toNT(ps₂, check_type)
check_type(A::AbstractMatrix) = @test typeof(A) <: StiefelManifold

check_type(ps₂)