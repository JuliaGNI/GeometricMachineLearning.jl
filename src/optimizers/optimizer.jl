
#######################################################################################
#Optimiser

struct Optimizer{MT<:OptimizerMethod, CT<:NamedTuple}
    method::MT
    cache::CT
    step::Int
end

function Optimizer(m::OptimizerMethod, x)
    Optimizer(m, init_optimizer_cache(m, x), 0)
end

#######################################################################################
#optimization step function

function optimization_step!(m::OptimizerMethod, d::Lux.AbstractExplicitLayer, ps::NamedTuple, C::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    λY = GlobalSection(ps)
    B = global_rep(λY, gx)
    update!(m, C, B)
    ps₂ = retraction(d, B)
    apply_section!(ps, λY, ps₂)
end

function optimization_step!(o::Optimizer, model::Lux.Chain, ps::NamedTuple, dx::NamedTuple)
    o.step += 1
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

function update!(m::OptimizerMethod, C::NamedTuple, B::NamedTuple)
    apply_toNT(m, C, B, update!)
end

function apply_toNT(m::OptimizerMethod, ps₁::NamedTuple, ps₂::NamedTuple, fun_name)    
    apply_toNT(ps₁, ps₂, (ps₁, ps₂) -> fun(m, ps₁, ps₂))
end
