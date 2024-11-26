@doc raw"""
    AbstractCache

`AbstractCache` has subtypes: [`AdamCache`](@ref), [`MomentumCache`](@ref), [`GradientCache`](@ref) and [`BFGSCache`](@ref).

All of them can be initialized with providing an array (also supporting manifold types).
"""
abstract type AbstractCache{T} end

#############################################################################
# All the definitions of the caches

@doc raw"""
    AdamCache(Y)

Store the first and second moment for `Y` (initialized as zeros).

First and second moments are called `B₁` and `B₂`.

If the cache is called with an instance of a homogeneous space, e.g. the [`StiefelManifold`](@ref) ``St(n,N)`` it initializes the moments as elements of ``\mathfrak{g}^\mathrm{hor}`` ([`StiefelLieAlgHorMatrix`](@ref)).

# Examples

```jldoctest
using GeometricMachineLearning

Y = rand(StiefelManifold, 5, 3)
AdamCache(Y).B₁

# output

5×5 StiefelLieAlgHorMatrix{Float64, SkewSymMatrix{Float64, Vector{Float64}}, Matrix{Float64}}:
 0.0  -0.0  -0.0  -0.0  -0.0
 0.0   0.0  -0.0  -0.0  -0.0
 0.0   0.0   0.0  -0.0  -0.0
 0.0   0.0   0.0   0.0   0.0
 0.0   0.0   0.0   0.0   0.0
```
"""
struct AdamCache{T, AT <: AbstractArray{T}} <: AbstractCache{T}
    B₁::AT
    B₂::AT 
    function AdamCache(Y::AbstractArray)
        new{eltype(Y), typeof(zero(Y))}(zero(Y), zero(Y))
    end
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, C::AdamCache)
    println(io, raw"`AdamCache` that currently stores `B₁` as ...")
    show(io, "text/plain", C.B₁)
    println(io, "")
    println(io, raw"and `B₂` as ...")
    show(io, "text/plain", C.B₂)
end

@doc raw"""
    MomentumCache(Y)

Store the moment for `Y` (initialized as zeros).

The moment is called `B`.

If the cache is called with an instance of a [`Manifold`](@ref) it initializes the moments as elements of ``\mathfrak{g}^\mathrm{hor}`` ([`AbstractLieAlgHorMatrix`](@ref)).

See [`AdamCache`](@ref).
"""
struct MomentumCache{T, AT <: AbstractArray{T}} <:AbstractCache{T}
    B::AT
    function MomentumCache(Y::AbstractArray)
        new{eltype(Y), typeof(zero(Y))}(zero(Y))
    end
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, C::MomentumCache)
    println(io, raw"`MomentumCache` that currently stores `B`as  ...")
    show(io, "text/plain", C.B)
end

@doc raw"""
    GradientCache(Y)

Do not store anything.

The cache for the [`GradientOptimizer`](@ref) does not consider past information.
"""
struct GradientCache{T} <: AbstractCache{T} end
GradientCache(::AbstractArray{T}) where T = GradientCache{T}()

#############################################################################
# All the setup_cache functions 

setup_adam_cache(ps::NamedTuple) = apply_toNT(setup_adam_cache, ps)
setup_momentum_cache(ps::NamedTuple) = apply_toNT(setup_momentum_cache, ps)
setup_gradient_cache(ps::NamedTuple) = apply_toNT(setup_gradient_cache, ps)

function setup_cache(_setup_cache_function, ps::NeuralNetworkParameters)
    ps_keys = keys(ps)
    values = Tuple([_setup_cache_function(ps[key]) for key in ps_keys])
    NamedTuple{ps_keys}(values)
end

setup_adam_cache(ps::NeuralNetworkParameters) = setup_cache(setup_adam_cache, ps)
setup_momentum_cache(ps::NeuralNetworkParameters) = setup_cache(setup_momentum_cache, ps)
setup_gradient_cache(ps::NeuralNetworkParameters) = setup_cache(setup_gradient_cache, ps)

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
