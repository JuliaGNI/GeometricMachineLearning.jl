"""
Optimizer struct that stores the 'method' (i.e. Adam with corresponding hyperparameters), the cache and the optimization step.

It takes as input an optimization method and the parameters of a network. 
"""
mutable struct Optimizer{MT<:OptimizerMethod, CT}
    method::MT
    cache::CT
    step::Int
end

function Optimizer(m::OptimizerMethod, x)
    Optimizer(m, init_optimizer_cache(m, x), 0)
end


#######################################################################################
# optimization step function

function optimization_step!(o::Optimizer, d::Union{AbstractExplicitLayer, AbstractExplicitCell}, ps::NamedTuple, C::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    λY = GlobalSection(ps)
    B = global_rep(λY, gx)
    update!(o, C, B)
    ps₂ = retraction(d, B)
    apply_section!(ps, λY, ps₂)
end

function optimization_step!(o::Optimizer, model::Chain, ps, dx)
    o.step += 1
    for (index, element) in zip(eachindex(model.layers), model)
        optimization_step!(o, element, ps[index...], o.cache[index...], dx[index...])
    end
end

function optimization_step!(o::Optimizer, model::AbstractExplicitLayer, ps, dx)
    o.step += 1
    optimization_step!(o, model, ps, o.cache[1], dx)
end


#######################################################################################
# utils functions (should probably be put somewhere else)

rgrad(ps::NamedTuple, dx::NamedTuple) = apply_toNT(rgrad, ps, dx)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end

function update!(m::Optimizer, C::NamedTuple, B::NamedTuple)
    apply_toNT(m, C, B, update!)
end

function apply_toNT(m::Optimizer, ps₁::NamedTuple, ps₂::NamedTuple, fun_name)    
    apply_toNT((ps₁, ps₂) -> fun_name(m, ps₁, ps₂), ps₁, ps₂)
end
