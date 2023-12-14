@doc raw"""
AbstractCache has subtypes: 
- AdamCache
- MomentumCache
- GradientCache
- BFGSCache

All of them can be initialized with providing an array (also supporting manifold types).
"""
abstract type AbstractCache{T} end

#############################################################################
# All the definitions of the caches

struct AdamCache{T, AT <: AbstractArray{T}} <: AbstractCache{T}
    B₁::AT
    B₂::AT 
    function AdamCache(Y::AbstractArray)
        new{eltype(Y), typeof(zero(Y))}(zero(Y), zero(Y))
    end
end

struct MomentumCache{T, AT <: AbstractArray{T}} <:AbstractCache{T}
    B::AT
    function MomentumCache(Y::AbstractArray)
        new{eltype(Y), typeof(zero(Y))}(zero(Y))
    end
end

struct GradientCache{T} <: AbstractCache{T} end
GradientCache(::AbstractArray{T}) where T = GradientCache{T}()

#############################################################################
# All the setup_cache functions 

# I don't really understand what we need these for ???
# setup_adam_cache(B::AbstractArray) = reshape([setup_adam_cache(b) for b in B], size(B))
# setup_momentum_cache(B::AbstractArray) = reshape([setup_momentum_cache(b) for b in B], size(B))
# setup_gradient_cache(B::AbstractArray) = reshape([setup_gradient_cache(b) for b in B], size(B))

setup_adam_cache(ps::NamedTuple) = apply_toNT(setup_adam_cache, ps)
setup_momentum_cache(ps::NamedTuple) = apply_toNT(setup_momentum_cache, ps)
setup_gradient_cache(ps::NamedTuple) = apply_toNT(setup_gradient_cache, ps)

setup_adam_cache(ps::Tuple) = Tuple([setup_adam_cache(x) for x in ps])
setup_momentum_cache(ps::Tuple) = Tuple([setup_momentum_cache(x) for x in ps])
setup_gradient_cache(ps::Tuple) = Tuple([setup_gradient_cache(x) for x in ps])

setup_adam_cache(B::AbstractArray{<:Number}) = AdamCache(B)
setup_momentum_cache(B::AbstractArray{<:Number}) = MomentumCache(B)
setup_gradient_cache(B::AbstractArray{<:Number}) = GradientCache(B)

function Base.zero(Y::StiefelManifold{T}) where T 
    N, n = size(Y)
    backend = KernelAbstractions.get_backend(Y.A)
    zeros(backend, StiefelLieAlgHorMatrix{T}, N, n)
end

function Base.zero(Y::GrassmannManifold{T}) where T 
    N, n = size(Y)
    backend = KernelAbstractions.get_backend(Y.A)
    zeros(backend, GrassmannLieAlgHorMatrix{T}, N, n)
end
