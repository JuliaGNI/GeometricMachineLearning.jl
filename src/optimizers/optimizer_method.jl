@doc raw"""
Each `Optimizer` has to be called with an `OptimizerMethod`. This specifies how the neural network weights are updated in each optimization step.
"""
abstract type OptimizerMethod end

@doc raw"""
    init_optimizer_cache(method, x)

Initialize the optimizer cache based on input `x` for the given `method`.
"""
function init_optimizer_cache(::OptimizerMethod, x) end

@doc raw"""
    update!(o, cache, dx::AbstractArray)

Update the `cache` based on the gradient information `dx`, compute the final velocity and store it in `dx`.

The optimizer `o` is needed because some updating schemes (such as [`AdamOptimizer`](@ref)) also need information on the current time step.
"""
function update!(::Any, ::Any, ::AbstractArray) end