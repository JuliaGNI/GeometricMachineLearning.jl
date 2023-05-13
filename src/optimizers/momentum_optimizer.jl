"""
Define the Momentum optimizer, i.e. 
V ← α*V - ∇f(W)
W ← W + η*V
Or the riemannian manifold equivalent, if applicable.
"""
mutable struct MomentumOptimizer{T} <: AbstractOptimizer
    η::T
    α::T
    t::Int
    MomentumOptimizer(η = 1e-3, α = 1e-2) = new{typeof(η)}(η, α, 0)
end

#update for single layer
function update!(o::MomentumOptimizer, C::MomentumLayerCache, B::NamedTuple)
    #o.t += 1
    for key in keys(B)
        C.B[key] .= α*C.B[key] + B[key]
        B[key] .= -o.η*C.B[key]
    end 
    B
end
