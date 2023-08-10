"""
Define the Momentum optimizer, i.e. 
V ← α*V - ∇f(W)
W ← W + η*V
Or the riemannian manifold equivalent, if applicable.
"""
struct MomentumOptimizer{T<:Real} <: OptimizerMethod
    η::T
    α::T
    MomentumOptimizer(η = 1e-3, α = 1e-2) = new{typeof(η)}(η, α)
end

#update for weights
function update!(o::Optimizer{<:MomentumOptimizer}, C::MomentumCache, B::AbstractVecOrMat)
    add!(C.B, o.method.α*C.B, B)
    mul!(B, -o.method.η, C.B)
end