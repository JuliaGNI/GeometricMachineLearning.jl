#This files contains Cache's structure

abstract type AbstractCache end

#############################################################################
# All the definitions of the caches

mutable struct AdamCache{T, AT <: AbstractArray{T}} <: AbstractCache
    B₁::AT
    B₂::AT 
    function AdamCache(B::AbstractArray)
        new{eltype(B), typeof(similar(B))}(zero(B), zero(B))
    end
end

mutable struct MomentumCache{T, AT <: AbstractArray{T}} <:AbstractCache
    B::AT
    function MomentumCache(B::AbstractArray)
        new{eltype(B), typeof(B)}(similar(B))
    end
end

struct GradientCache <: AbstractCache end
GradientCache(::AbstractMatrix) = GradientCache()

#############################################################################
# All the setup_cache functions 

setup_adam_cache(dx::NamedTuple) = apply_toNT(dx, setup_adam_cache)
setup_momentum_cache(dx::NamedTuple) = apply_toNT(dx, setup_momentum_cache)
setup_gradient_cache(dx::NamedTuple) = apply_toNT(dx, setup_gradient_cache)

setup_adam_cache(B::AbstractMatrix) = AdamCache(B)
setup_momentum_cache(B::AbstractMatrix) = MomentumCache(B)
setup_gradient_cache(B::AbstractMatrix) = GradientCache(B)


function Base.similar(Y::StiefelManifold{T}) where T 
    N, n = size(Y)
    zeros(StiefelLieAlgHorMatrix{T}, N, n)
end

function Base.zero(Y::StiefelManifold{T}) where T 
    N, n = size(Y)
    zeros(StiefelLieAlgHorMatrix{T}, N, n)
end