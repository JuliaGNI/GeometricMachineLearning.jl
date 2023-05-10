
abstract type AbstractOptimizer end

function optimization_step!(o::AbstractOptimizer, d::Lux.AbstractExplicitLayer, ps::NamedTuple, dx::NamedTuple)
    gx = rgrad(d, ps, dx)
    λY = GlobalSection(d, ps)
    B = globalrep(d, λY, gx)
    update!(o, C, B)
    ps = retraction(d, B)
    apply(λY, ps)
end

function optimization_step!(o::AbstractOptimizer, model::Lux.Chain, ps::NamedTuple, dx::NamedTuple)
    o.t += 1
    i = 0
    for key in keys(model)
        i += 1
        optimization_step(o, model[i], ps[key], dx[key])
    end
end