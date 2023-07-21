#This files contains Cache's structure

abstract type AbstractCache end

#############################################################################
# All the definitions of the caches

mutable struct AdamCache{T, AT <: AbstractArray{T}} <: AbstractCache
    B₁::AT
    B₂::AT 
    function AdamCache(B::AbstractArray)
        new{eltype(B), typeof(B)}(similar(B), similar(B))
    end
end

mutable struct MomentumCache{T, AT <: AbstractArray{T}} <:AbstractCache
    B::AT
    function MomentumCache(B::AbstractArray)
        new{eltype(B), typeof(B)}(similar(B))
    end
end

struct GradientCache <: AbstractCache end
GradientCache(::AbstractArray) = GradientCache()

#############################################################################
# All the setup_cache functions 

setup_adam_cache(dx::NamedTuple) = apply_toNT(setup_adam_cache, dx)
setup_momentum_cache(dx::NamedTuple) = apply_toNT(setup_momentum_cache, dx)
setup_gradient_cache(dx::NamedTuple) = apply_toNT(setup_gradient_cache, dx)

setup_adam_cache(dx::Tuple) = Tuple([setup_adam_cache(x) for x in dx])
setup_momentum_cache(dx::Tuple) = Tuple([setup_momentum_cache(x) for x in dx])
setup_gradient_cache(dx::Tuple) = Tuple([setup_gradient_cache(x) for x in dx])

setup_adam_cache(B::AbstractArray) = AdamCache(B)
setup_momentum_cache(B::AbstractArray) = MomentumCache(B)
setup_gradient_cache(B::AbstractArray) = GradientCache(B)


