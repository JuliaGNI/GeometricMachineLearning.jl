@doc raw"""
    Optimizer(method, cache, step)

Store the `method` (e.g. [`AdamOptimizer`](@ref) with corresponding hyperparameters), the `cache` (e.g. [`AdamCache`](@ref)) and the optimization step.

It takes as input an optimization method and the parameters of a network. 

For *technical reasons* we first specify an [`OptimizerMethod`](@ref) that stores all the hyperparameters of the optimizer. 
"""
mutable struct Optimizer{MT<:OptimizerMethod, CT}
    method::MT
    cache::CT
    step::Int
end

@doc raw"""
    Optimizer(method, nn_params)

Allocate the cache for a specific `method` and `nn_params` for an instance of `Optimizer`.

Internally this calls [`init_optimizer_cache`](@ref).
"""
function Optimizer(method::OptimizerMethod, nn_params::Union{Tuple, NamedTuple})
    Optimizer(method, init_optimizer_cache(method, nn_params), 0)
end

"""
    Optimizer(method, nn::NeuralNetwork)

Allocate the cache for a specific `method` and a `NeuralNetwork` for an instance of `Optimizer`.

Internally this calls `Optimizer(method, nn.params)`.

Typically the Optimizer is not initialized with the network parameters, but instead with a NeuralNetwork struct.
"""
function Optimizer(method::OptimizerMethod, nn::NeuralNetwork)
    Optimizer(method, nn.params)
end

Optimizer(nn::NeuralNetwork, m::OptimizerMethod) = Optimizer(m, nn)

@doc raw"""
    update!(o, cache, B)

First update the `cache` and then update the array `B` based on the optimizer `o`. 

Note that ``B\in\mathfrak{g}^\mathrm{hor}`` in general.
"""
function update!(::Optimizer, ::AbstractCache, ::AbstractArray) end

#######################################################################################
# optimization step function

@doc raw"""
    optimization_step!(o, layer, ps, cache, dx)

Update the weights `ps` of a `layer` based on an [`Optimizer`](@ref), a `cache` and first-order derivatives `dx`.

The derivatives `dx` here are usually obtained via an AD routine by differentiating a loss function, i.e. `dx` is ``\nabla_xL``.

It is calling the function [`update!`](@ref) internally which has to be implemented for every [`OptimizerMethod`](@ref).
"""
function optimization_step!(o::Optimizer, layer::Union{AbstractExplicitLayer, AbstractExplicitCell}, ps::NamedTuple, cache::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    λY = GlobalSection(ps)
    B = global_rep(λY, gx)
    update!(o, cache, B)
    ps₂ = retraction(layer, B)
    apply_section!(ps, λY, ps₂)
end

@doc raw"""
    optimization_step!(o::Optimizer, model::Chain, ps::Tuple, dx::Tuple)

Optimize a neural network built with `Chain`.
"""
function optimization_step!(o::Optimizer, model::Chain, ps::Tuple, dx::Tuple)
    o.step += 1
    for (index, element) in zip(eachindex(model.layers), model.layers)
        optimization_step!(o, element, ps[index], o.cache[index], dx[index])
    end
end

@doc raw"""
    optimization_step!(o::Optimizer, model::AbstractExplicitLayer, ps::NamedTuple, dx::NamedTuple)

Optimize a neural network consisting of a single `AbstractExplicitLayer`.
"""
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
