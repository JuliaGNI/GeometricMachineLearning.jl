
abstract type AbstractOptimizer end

function optimization_step!(o::AbstractOptimizer, d::Lux.AbstractExplicitLayer, ps::NamedTuple, C::AbstractLayerCache, dx::NamedTuple)
    gx = rgrad(d, ps, dx)
    λY = GlobalSection(d, ps)
    B = global_rep(d, λY, gx)
    update!(o, C, B)
    ps = retraction(d, B)
    apply(λY, ps)
end

function optimization_step!(o::AbstractOptimizer, model::Lux.Chain, ps::NamedTuple, cache::NamedTuple, dx::NamedTuple)
    o.t += 1
    i = 0
    for key in keys(model)
        i += 1
        optimization_step!(o, model[i], ps[key], cache[key], dx[key])
    end
end

#add a routine that can deal with a single layer (no Lux.Chain)
#function optimization_step!(o::AbstractOptimizer, model::Lux.AbstractExplicitLayer, ps::NamedTuple, cache::NamedTuple, dx::NamedTuple)
#end

function optimization_step!(o::AbstractOptimizer, model::Lux.Chain, ps::NamedTuple, loss)
    dx = Zygote.gradient(ps -> loss(ps), ps)[1]
    optimization_step!(o, model, ps, dx)
end 

rgrad(d::Lux.AbstractExplicitLayer, ps::NamedTuple, dx::NamedTuple) = dx