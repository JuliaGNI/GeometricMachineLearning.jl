
#######################################################################################
#Optimiser

struct Optimizer{MT<:AbstractMethodOptimiser, CT<:NamedTuple}
    method::MT
    cache::CT

    function Optimizer(m::AbstractMethodOptimiser, model::Lux.Chain)
        cache = init_optimizer_cache(m, model)
        new{typeof(m),typeof(cache)}(m,cache)
    end
end

#######################################################################################
#init_cache

"""
This initializes the cache for the entire chain
"""

layer_cache(::GradientOptimizer, d::Lux.AbstractExplicitLayer) = StandardLayerCache(d)  #GradientLayerCache
layer_cache(::MomentumOptimizer, d::Lux.AbstractExplicitLayer) = MomentumLayerCache(d)
layer_cache(::AdamOptimizer, d::Lux.AbstractExplicitLayer) = AdamLayerCache(d)

function init_optimizer_cache(m::AbstractMethodOptimiser, model::Lux.Chain)
    layers = keys(model)
    cache = NamedTuple()
    i = 0
    for layer in layers
         i += 1
        layer_cacheᵢ = layer_cache(m, model[i])
        cache = merge(cache, 
                NamedTuple{(layer, )}((layer_cacheᵢ,))
                )
    end
    cache
end

#######################################################################################
#optimization step function

function optimization_step!(m::AbstractMethodOptimiser, d::Lux.AbstractExplicitLayer, ps::NamedTuple, C::AbstractLayerCache, dx::NamedTuple)
    gx = rgrad(d, ps, dx)
    λY = GlobalSection(d, ps)
    B = global_rep(d, λY, gx)
    update!(m, C, B)
    ps = retraction(d, B)
    apply(λY, ps)
end

function optimization_step!(o::Optimizer, model::Lux.Chain, ps::NamedTuple, dx::NamedTuple)
    o.method.t += 1
    i = 0
    for key in keys(model)
        i += 1
        optimization_step!(o.method, model[i], ps[key], o.cache[key], dx[key])
    end
end

function optimization_step!(o::Optimizer, model::Lux.Chain, ps::NamedTuple, loss)
    dx = Zygote.gradient(ps -> loss(ps), ps)[1]
    optimization_step!(o, model, ps, dx)
end 

