function optimizer_cache(d::Lux.Chain, o::AbstracOptimizer, B::NamedTuple)
    layers = keys(B)
    cache = NamedTuple()
    i = 0
    for layer in layers
         i += 1
        layer_cache = layer_cache(d[i], o, B[key])
        merge(cache, NamedTuple{(layer, )}(layer_cache))
        end
    end
    cache
end