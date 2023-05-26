
abstract type AbstractOptimizer end


function optimization_step!(o::AbstractOptimizer, d::Lux.AbstractExplicitLayer, ps::NamedTuple, C::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    λY = GlobalSection(ps)
    B = global_rep(λY, gx)
    update!(o, C, B)
    ps₂ = retraction(d, B)
    apply_section!(ps, λY, ps₂)
end

function optimization_step!(o::AbstractOptimizer, model::Lux.Chain, ps::NamedTuple, C::NamedTuple, dx::NamedTuple)
    o.t += 1
    i = 0
    for key in keys(model)
        i += 1
        optimization_step!(o, model[i], ps[key], C[key], dx[key])
    end
end

#add a routine that can deal with a single layer (no Lux.Chain)
#function optimization_step!(o::AbstractOptimizer, model::Lux.AbstractExplicitLayer, ps::NamedTuple, cache::NamedTuple, dx::NamedTuple)
#end

function optimization_step!(o::AbstractOptimizer, model::Lux.Chain, ps::NamedTuple, loss)
    dx = Zygote.gradient(ps -> loss(ps), ps)[1]
    optimization_step!(o, model, ps, dx)
end 

rgrad(ps::NamedTuple, dx::NamedTuple) = apply_toNT(ps, dx, rgrad)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end

function apply_toNT(o::AbstractOptimizer, ps₁::NamedTuple, ps₂::NamedTuple, fun_name)    
    keys₁ = keys(ps₁)
    @assert keys₁ == keys(ps₂)
    ps_applied = NamedTuple()
    for key in keys(ps₁)
        ps_applied = merge(ps_applied, NamedTuple{(key, )}((fun_name(o, ps₁[key], ps₂[key]), )))
    end
    ps_applied
end

function update!(o::AbstractOptimizer, C::NamedTuple, B::NamedTuple)
    apply_toNT(o, C, B, update!)
end
