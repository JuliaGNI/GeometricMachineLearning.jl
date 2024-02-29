@doc raw"""
Optimizer struct that stores the 'method' (i.e. Adam with corresponding hyperparameters), the cache and the optimization step.

It takes as input an optimization method and the parameters of a network. 

For *technical reasons* we first specify an OptimizerMethod that stores all the hyperparameters of the optimizer. 
"""
mutable struct Optimizer{MT<:OptimizerMethod, CT}
    method::MT
    cache::CT
    step::Int
end

function Optimizer(m::OptimizerMethod, x::Union{Tuple, NamedTuple})
    Optimizer(m, init_optimizer_cache(m, x), 0)
end

"""
Typically the Optimizer is not initialized with the network parameters, but instead with a NeuralNetwork struct.
"""
function Optimizer(m::OptimizerMethod, nn::NeuralNetwork)
    Optimizer(m, nn.params)
end

Optimizer(nn::NeuralNetwork, m::OptimizerMethod) = Optimizer(m, nn)

#######################################################################################
# optimization step function

@doc raw"""
Optimization for a single layer. 

inputs: 
- `o::Optimizer`
- `d::Union{AbstractExplicitLayer, AbstractExplicitCell}`
- `ps::NamedTuple`: the parameters 
- `C::NamedTuple`: NamedTuple of the caches 
- `dx::NamedTuple`: NamedTuple of the derivatives (output of AD routine)

`ps`, `C` and `dx` must have the same keys. 
"""
function optimization_step!(o::Optimizer, d::Union{AbstractExplicitLayer, AbstractExplicitCell}, ps::NamedTuple, C::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    λY = GlobalSection(ps)
    B = global_rep(λY, gx)
    update!(o, C, B)
    ps₂ = retraction(d, B)
    apply_section!(ps, λY, ps₂)
end

@doc raw"""
Optimization for an entire neural network, the way this function should be called. 

inputs: 
- `o::Optimizer`
- `model::Chain`
- `ps::Tuple`
- `dx::Tuple`
"""
function optimization_step!(o::Optimizer, model::Chain, ps::Tuple, dx::Tuple)
    o.step += 1
    for (index, element) in zip(eachindex(model.layers), model.layers)
        optimization_step!(o, element, ps[index], o.cache[index], dx[index])
    end
end

function optimization_step!(o::Optimizer, model::AbstractExplicitLayer, ps::NamedTuple, dx::NamedTuple)
    o.step += 1

    optimization_step!(o, model, ps, o.cache, dx)
end

#######################################################################################
# utils functions (should probably be put somewhere else)

rgrad(ps::NamedTuple, dx::NamedTuple) = apply_toNT(rgrad, ps, dx)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end

# do we need those two? 
function update!(m::Optimizer, C::NamedTuple, B::NamedTuple)
    apply_toNT(m, C, B, update!)
end

function apply_toNT(m::Optimizer, ps₁::NamedTuple, ps₂::NamedTuple, fun_name)    
    apply_toNT((ps₁, ps₂) -> fun_name(m, ps₁, ps₂), ps₁, ps₂)
end
