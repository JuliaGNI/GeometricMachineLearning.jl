"""
This initializes the cache for the entire chain
"""

layer_cache(::StandardOptimizer, d::Lux.AbstractExplicitLayer) = StandardLayerCache(d)
layer_cache(::MomentumOptimizer, d::Lux.AbstractExplicitLayer) = MomentumLayerCache(d)
layer_cache(::AdamOptimizer, d::Lux.AbstractExplicitLayer) = AdamLayerCache(d)

function init_optimizer_cache(model::Lux.Chain, o::AbstractOptimizer)
    layers = keys(model)
    cache = NamedTuple()
    i = 0
    for layer in layers
         i += 1
        layer_cacheᵢ = layer_cache(o, model[i])
        cache = merge(cache, 
                NamedTuple{(layer, )}((layer_cacheᵢ,))
                )
    end
    cache
end

#add a routine to make this work for single layers (not just Lux.Chains)