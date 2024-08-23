@doc raw"""
    Optimizer(method, cache, step, retraction)

Store the `method` (e.g. [`AdamOptimizer`](@ref) with corresponding hyperparameters), the `cache` (e.g. [`AdamCache`](@ref)), the optimization step and the retraction.

It takes as input an optimization method and the parameters of a network. 

Before one can call `Optimizer` a [`OptimizerMethod`](@ref) that stores all the hyperparameters of the optimizer needs to be specified. 

# Functor 

For an instance `o` of `Optimizer`, we can call the corresponding functor as:

```julia
o(nn, dl, batch, n_epochs, loss)
```

The arguments are:
1. `nn::NeuralNetwork`
2. `dl::`[`DataLoader`](@ref)
3. `batch::`[`Batch`](@ref)
4. `n_epochs::Integer`
5. `loss::`[`NetworkLoss`](@ref)

The last argument is optional for many neural network architectures. We have the following defaults:
- A [`TransformerIntegrator`](@ref) uses [`TransformerLoss`](@ref).
- A [`NeuralNetworkIntegrator`](@ref) uses [`FeedForwardLoss`](@ref).
- An [`AutoEncoder`](@ref) uses [`AutoEncoderLoss`](@ref).

In addition there is an optional keyword argument that can be supplied to the functor:
- `show_progress=true`: This specifies whether a progress bar should be shown during training.

# Implementation

Internally the functor for `Optimizer` calls [`GlobalSection`](@ref) once at the start and then [`optimize_for_one_epoch!`](@ref) for each epoch.
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

An equivalent constructor is

```julia
Optimizer(method, nn::NeuralNetwork)
```

# Arguments

The optional keyword argument is the retraction. By default this is [`cayley`](@ref).
"""
function Optimizer(method::OptimizerMethod, nn_params::Union{Tuple, NamedTuple}; retraction = cayley)
    Optimizer(method, init_optimizer_cache(method, nn_params), 0, retraction)
end

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

function _optimization_step!(o::Optimizer, λY::NamedTuple, ps::NamedTuple, cache::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    B = global_rep(λY, gx)
    update!(o, cache, B)
    update_section!(λY, B, o.retraction)

    nothing
end

function optimization_step!(o::Optimizer, λY::Tuple, ps::Tuple, dx::Tuple)
    o.step += 1
    for (cache, λY, ps, dx) in zip(o.cache, λY, ps, dx)
        _optimization_step!(o, λY, ps, cache, dx)
    end
end

@doc raw"""
    optimization_step!(o, λY, ps, dx)

Update the weights `ps` based on an [`Optimizer`](@ref), a `cache` and first-order derivatives `dx`.

`optimization_step!` is calling [`update!`](@ref) internally. 
`update!` has to be implemented for every [`OptimizerMethod`](@ref).

# Arguments

All arguments into `optimization_step!` are mandatory:
1. `o::`[`Optimizer`](@ref),
2. `λY::NamedTuple`: this named tuple has the same keys as `ps`, but contains [`GlobalSection`](@ref)s,
3. `ps::NamedTuple`: the neural network parameters,
5. `dx::NamedTuple`: the gradients stores as a NamedTuple.

All the arguments are given as `NamedTuple`s  as the neural network weights are stores in that format.

```jldoctest
using GeometricMachineLearning

l = StiefelLayer(3, 5)
ps = initialparameters(l, Float32)
cache = apply_toNT(MomentumCache, ps)
o = Optimizer(MomentumOptimizer(), cache, 0, geodesic)
λY = GlobalSection(ps)
dx = (weight = rand(Float32, 5, 3), )

# call the optimizer
optimization_step!(o, λY, ps, dx)

_test_nt(x) = typeof(x) <: NamedTuple

_test_nt(λY) & _test_nt(ps) & _test_nt(cache) & _test_nt(dx)

# output

true
```

Note that we used `initialparameters` here instead of `NeuralNetwork` (as we do usually).

# Extended help
The derivatives `dx` here are usually obtained via an AD routine by differentiating a loss function, i.e. `dx` is ``\nabla_xL``.
"""
function optimization_step!(o::Optimizer, λY::NamedTuple, ps::NamedTuple, dx::NamedTuple)
    o.step += 1

    _optimization_step!(o, λY, ps, o.cache, dx)
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
