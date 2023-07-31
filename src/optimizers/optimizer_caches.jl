"""
AbstractCache has subtypes: 
AdamCache
MomentumCache
GradientCache

All of them can be initialized with providing an array (also supporting manifold types).
"""
abstract type AbstractCache end

#############################################################################
# All the definitions of the caches

struct AdamCache{T, AT <: AbstractArray{T}} <: AbstractCache
    B₁::AT
    B₂::AT 
    function AdamCache(B::AbstractArray)
        new{eltype(B), typeof(zero(B))}(zero(B), zero(B))
    end
end

struct MomentumCache{T, AT <: AbstractArray{T}} <:AbstractCache
    B::AT
    function MomentumCache(B::AbstractArray)
        new{eltype(B), typeof(zero(B))}(zero(B))
    end
end

struct GradientCache <: AbstractCache end
GradientCache(::AbstractArray) = GradientCache()

#############################################################################
# All the setup_cache functions 

setup_adam_cache(ps::NamedTuple) = apply_toNT(setup_adam_cache, ps)
setup_momentum_cache(ps::NamedTuple) = apply_toNT(setup_momentum_cache, ps)
setup_gradient_cache(ps::NamedTuple) = apply_toNT(setup_gradient_cache, ps)

setup_adam_cache(ps::Tuple) = Tuple([setup_adam_cache(x) for x in ps])
setup_momentum_cache(ps::Tuple) = Tuple([setup_momentum_cache(x) for x in ps])
setup_gradient_cache(ps::Tuple) = Tuple([setup_gradient_cache(x) for x in ps])

setup_adam_cache(B::AbstractArray) = AdamCache(B)
setup_momentum_cache(B::AbstractArray) = MomentumCache(B)
setup_gradient_cache(B::AbstractArray) = GradientCache(B)

function Base.zero(Y::StiefelManifold{T}) where T 
    N, n = size(Y)
    zeros(StiefelLieAlgHorMatrix{T}, N, n)
end

function Base.zero(Y::GrassmannManifold{T}) where T 
    N, n = size(Y)
    zeros(GrassmannLieAlgHorMatrix{T}, N, n)
end