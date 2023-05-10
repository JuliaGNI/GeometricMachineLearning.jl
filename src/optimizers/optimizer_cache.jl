function init_optimizer_cache(d::Lux.Chain, o::AbstracOptimizer)
    layers = keys(B)
    cache = NamedTuple()
    i = 0
    for layer in layers
         i += 1
        layer_cache = layer_cache(o, d[i])
        merge(cache, NamedTuple{(layer, )}(layer_cache))
        end
    end
    cache
end

layer_cache(::StandardOptimizer, d::Lux.AbstractExplicitLayer) = StandardLayerCache(d)
layer_cache(::MomentumOptimizer, d::Lux.AbstractExplicitLayer) = MomentumLayerCache(d)
layer_cache(::AdamOptimizer, d::Lux.AbstractExplicitLayer) = AdamLayerCache(d)