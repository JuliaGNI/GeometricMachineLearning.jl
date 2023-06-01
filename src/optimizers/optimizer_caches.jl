abstract type AbstractCache end

mutable struct AdamCache{T, AT <: AbstractMatrix} <: AbstractCache
    B₁::AT
    B₂::AT 
    function AdamCache(B₁::AbstractMatrix, B₂::AbstractMatrix)
        new{eltype(B₁), typeof(B₁)}(B₁, B₂)
    end
end

mutable struct MomentumCache{T, AT <: AbstractMatrix} <:AbstractCache
    B::AT
    function MomentumCache(B::AbstractMatrix)
        new{eltype(B), typeof(B)}(B)
    end
end

struct StandardCache <: AbstractCache end
StandardCache(::AbstractMatrix) = StandardCache()

function setup_adam_cache(B₁::NamedTuple, B₂::NamedTuple)
    apply_toNT(B₁, B₂, setup_adam_cache)
end

setup_adam_cache(B₁::AbstractMatrix, B₂::AbstractMatrix) = AdamCache(B₁, B₂)


function setup_momentum_cache(dx::NamedTuple)
    apply_toNT(dx, setup_momentum_cache)
end

setup_momentum_cache(B::AbstractMatrix) = MomentumCache(B)

function setup_standard_cache(dx::NamedTuple)
    apply_toNT(dx, setup_standard_cache)
end

setup_standard_cache(B::AbstractMatrix) = StandardCache(B)

function setup_adam_cache(d::Lux.AbstractExplicitLayer)
    B₁, _ = Lux.setup(TrivialInitRNG(), d) #.|> Lux.gpu
    B₂, _ = Lux.setup(TrivialInitRNG(), d) #.|> Lux.gpu
    setup_adam_cache(B₁, B₂)
end

function setup_momentum_cache(d::Lux.AbstractExplicitLayer)
    B, _ = Lux.setup(TrivialInitRNG(), d) #.|> Lux.gpu
    setup_momentum_cache(B)
end

#TODO: make this more efficient! you don't need to call the entire setup to initialize an empty NamedTuple!!!
function setup_standard_cache(d::Lux.AbstractExplicitLayer)
    B, _ = Lux.setup(TrivialInitRNG(), d) #.|> Lux.gpu
    setup_standard_cache(B)
end
