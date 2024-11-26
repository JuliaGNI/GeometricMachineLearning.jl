@doc raw"""
    OptimizerMethod

Each `Optimizer` has to be called with an `OptimizerMethod`. This specifies how the neural network weights are updated in each optimization step.
"""
abstract type OptimizerMethod{T} end

# @doc raw"""
#     init_optimizer_cache(method, x)
# 
# Initialize the optimizer cache based on input `x` for the given `method`.
# """
function init_optimizer_cache(::OptimizerMethod, x) end

eltype(::OptimizerMethod{T}) where T = T