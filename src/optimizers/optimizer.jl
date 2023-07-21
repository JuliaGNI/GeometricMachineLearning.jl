
#######################################################################################
#Optimiser

struct Optimizer{MT<:OptimizerMethod,CT<:Tuple}
    method::MT
    cache::CT
    step::Int
end

function Optimizer(m::OptimizerMethod, x)
    Optimizer(m, init_optimizer_cache(m, x), 0)
end

#######################################################################################
#optimization step function

function optimization_step!(m::OptimizerMethod, d::AbstractExplicitLayer, ps::NamedTuple, C::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    λY = GlobalSection(ps)
    B = global_rep(λY, gx)
    update!(m, C, B)
    ps₂ = retraction(d, B)
    apply_section!(ps, λY, ps₂)
end

function optimization_step!(o::Optimizer, model::Chain, ps::Tuple, dx::Tuple)
    o.step += 1
    for i in eachindex(model)
        optimization_step!(o.method, model[i], ps[i], o.cache[i], dx[i])
    end
end

function optimization_step!(o::Optimizer, model::Chain, ps::Tuple, loss)
    dx = Zygote.gradient(ps -> loss(ps), ps)[1]
    optimization_step!(o, model, ps, dx)
end 


#######################################################################################
#utils functions

rgrad(ps::NamedTuple, dx::NamedTuple) = apply_toNT(rgrad, ps, dx)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end

function update!(m::OptimizerMethod, C::NamedTuple, B::NamedTuple)
    apply_toNT(update!, m, C, B)
end

function apply_toNT(m::OptimizerMethod, ps₁::NamedTuple, ps₂::NamedTuple, fun_name)    
    apply_toNT((ps₁, ps₂) -> fun_name(m, ps₁, ps₂), ps₁, ps₂)
end
