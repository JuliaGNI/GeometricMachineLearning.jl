
#######################################################################################
#Optimiser

struct Optimizer{MT<:AbstractMethodOptimiser, CT<:NamedTuple}
    method::MT
    cache::CT

    function Optimizer(m::AbstractMethodOptimiser, model::Lux.AbstractExplicitLayer)
        cache = init_optimizer_cache(model, m)
        new{typeof(m),typeof(cache)}(m,cache)
    end
end


#######################################################################################
#optimization step function

function optimization_step!(m::AbstractMethodOptimiser, d::Lux.AbstractExplicitLayer, ps::NamedTuple, C::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    λY = GlobalSection(ps)
    B = global_rep(λY, gx)
    update!(m, C, B)
    ps₂ = retraction(d, B)
    apply_section!(ps, λY, ps₂)
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


#######################################################################################
#utils functions

rgrad(ps::NamedTuple, dx::NamedTuple) = apply_toNT(ps, dx, rgrad)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end

function update!(m::AbstractMethodOptimiser, C::NamedTuple, B::NamedTuple)
    apply_toNT(m, C, B, update!)
end

function apply_toNT(m::AbstractMethodOptimiser, ps₁::NamedTuple, ps₂::NamedTuple, fun_name)    
    keys₁ = keys(ps₁)
    @assert keys₁ == keys(ps₂)
    ps_applied = NamedTuple()
    for key in keys(ps₁)
        ps_applied = merge(ps_applied, NamedTuple{(key, )}((fun_name(m, ps₁[key], ps₂[key]), )))
    end
    ps_applied
end

