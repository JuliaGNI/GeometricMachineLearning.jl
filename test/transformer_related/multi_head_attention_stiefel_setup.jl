import Random, Test, Lux, LinearAlgebra, KernelAbstractions

using GeometricMachineLearning, Test
using GeometricMachineLearning: init_optimizer_cache

T = Float32
model = MultiHeadAttention(64, 8, Stiefel=true)
ps = initialparameters(KernelAbstractions.CPU(), T, model)
tol = 10*eps(T)

function check_setup(A::AbstractMatrix)
    @test typeof(A) <: StiefelManifold
    @test check(A) < tol
end
check_setup(ps::NamedTuple) = apply_toNT(check_setup, ps)
check_setup(ps)

######## check if the gradients are set up the correct way 
function check_grad_setup(B::AbstractMatrix)
    @test typeof(B) <: StiefelLieAlgHorMatrix
    @test LinearAlgebra.norm(B) < tol
end
check_grad_setup(gx::NamedTuple) = apply_toNT(check_grad_setup, gx)
check_grad_setup(B::MomentumCache) = check_grad_setup(B.B)

gx = init_optimizer_cache(MomentumOptimizer(), ps)
check_grad_setup(gx)
