@doc raw"""
    Optimizer(method, cache, step, retraction)

Store the `method` (e.g. [`AdamOptimizer`](@ref) with corresponding hyperparameters), the `cache` (e.g. [`AdamCache`](@ref)), the optimization step and the retraction.

It takes as input an optimization method and the parameters of a network. 

For *technical reasons* we first specify an [`OptimizerMethod`](@ref) that stores all the hyperparameters of the optimizer. 

# Functor 

```julia
Optimizer(nn, dl, batch, n_epochs, loss)
```

The arguments are the following
1. `nn::NeuralNetwork`
2. `dl::`[`DataLoader`](@ref)
3. `batch::`[`Batch`](@ref)
4. `n_epochs::Int`
5. `loss::`[`NetworkLoss`](@ref)

The last argument is optional for many neural network architectures. We have the following defaults:
- A [`TransformerIntegrator`](@ref) uses [`TransformerLoss`](@ref).
- A [`NeuralNetworkIntegrator`](@ref) uses [`FeedForwardLoss`](@ref).
- An [`AutoEncoder`](@ref) uses [`AutoEncoderLoss`](@ref).

In addition there is an optional keyword argument:
- `show_progress=true`: This specifies whether a progress bar should be shown during training.

# Implementation

Internally the functor for `Optimizer` calls [`GlobalSection`](@ref) and [`optimize_for_one_epoch!`](@ref).
"""
mutable struct Optimizer{MT<:OptimizerMethod, CT, RT}
    method::MT
    cache::CT
    step::Int
    retraction::RT
end

@doc raw"""
    Optimizer(method, nn_params)

Allocate the cache for a specific `method` and `nn_params` for an instance of `Optimizer`.

Internally this calls [`init_optimizer_cache`](@ref).

# Arguments

The optional keyword argument is the retraction. By default this is [`cayley`](@ref).
"""
function Optimizer(method::OptimizerMethod, nn_params::Union{Tuple, NamedTuple}; retraction = cayley)
    Optimizer(method, init_optimizer_cache(method, nn_params), 0, retraction)
end

"""
    Optimizer(method, nn::NeuralNetwork)

Allocate the cache for a specific `method` and a `NeuralNetwork` for an instance of `Optimizer`.

Internally this calls `Optimizer(method, nn.params)`.

Typically the Optimizer is not initialized with the network parameters, but instead with a NeuralNetwork struct.

# Arguments

See [`Optimizer(::OptimizerMethod, ::Union{Tuple, NamedTuple})`](@ref).
"""
function Optimizer(method::OptimizerMethod, nn::NeuralNetwork; kwargs...)
    Optimizer(method, nn.params; kwargs...)
end

Optimizer(nn::NeuralNetwork, m::OptimizerMethod; kwargs...) = Optimizer(m, nn; kwargs...)

@doc raw"""
    update!(o, cache, B)

First update the `cache` and then update the array `B` based on the optimizer `o`. 

Note that ``B\in\mathfrak{g}^\mathrm{hor}`` in general.
"""
function update!(::Optimizer, ::AbstractCache, ::AbstractArray) end

#######################################################################################
# optimization step function

@doc raw"""
    optimization_step!(o, λY, ps, cache, dx)

Update the weights `ps` of a `layer` based on an [`Optimizer`](@ref), a `cache` and first-order derivatives `dx`.

The derivatives `dx` here are usually obtained via an AD routine by differentiating a loss function, i.e. `dx` is ``\nabla_xL``.

It is calling the function [`update!`](@ref) internally which has to be implemented for every [`OptimizerMethod`](@ref).
"""
function optimization_step!(o::Optimizer, λY::NamedTuple, ps::NamedTuple, cache::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    B = global_rep(λY, gx)
    update!(o, cache, B)
    update_section!(λY, B, o.retraction)

    nothing
end

@doc raw"""
    optimization_step!(o::Optimizer, λY::Chain, ps::Tuple, dx::Tuple)

Optimize a neural network built with `Chain`.
"""
function optimization_step!(o::Optimizer, λY::Tuple, ps::Tuple, dx::Tuple)
    o.step += 1
    for (cache, λY, ps, dx) in zip(o.cache, λY, ps, dx)
        optimization_step!(o, λY, ps, cache, dx)
    end
end

@doc raw"""
    optimization_step!(o::Optimizer, λY::NamedTuple, ps::NamedTuple, dx::NamedTuple)

Optimize a neural network consisting of a single `AbstractExplicitLayer`.
"""
function optimization_step!(o::Optimizer, λY::NamedTuple, ps::NamedTuple, dx::NamedTuple)
    o.step += 1

    optimization_step!(o, λY, ps, o.cache, dx)
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
