@doc raw"""
This is an implementation of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer. 
"""
struct BFGSOptimizer{T<:Real} <: OptimizerMethod
    η::T

    BFGSOptimizer(η = 1f-2) = new{typeof(η)}(η)
end

@doc raw"""
Optimization for an entire neural networks with BFGS. What is different in this case is that we still have to initialize the cache. This was first done in a lazy way with `DummyBFGSCache`. 

If `o.step == 0`, then we initialize the cache
"""
function update!(o::Optimizer{<:BFGSOptimizer}, C::CT, B::AbstractArray{T}) where {T, CT<:BFGSCache{T}}
    if o.step == 0
        bfgs_initialization!(o, C, B)
    else
        bfgs_update!(o, C, B)   
    end
    o.step += 1
end

function bfgs_update!(o::Optimizer{<:BFGSOptimizer}, C::CT, B::AbstractArray{T}) where {T, CT<:BFGSCache{T}}
    # in the first step we compute the difference between the current and the previous mapped gradients:
    Y = B - C.B 
    # in the second step we update H (write a test to check that this preserves symmetry)
    vecS = vec(C.S)
    C.H .= C.H + (vecS' * vec(Y) + vec(Y)' * C.H * vec(Y)) / (vecS' * vec(Y)) ^ 2 * vecS * vecS' - (C.H * vec(Y) * vecS' + vecS * (C.H * vec(Y))' ) / (vecS' * vec(Y))
    # in the third step we compute the final velocity
    mul!(vecS, C.H, vec(B))
    mul!(C.S, C.S, -o.method.η)
    assign!(C.B, copy(B))
    assign!(B, copy(C.S))
end

function bfgs_initialization!(o::Optimizer{<:BFGSOptimizer}, C::CT, B::AbstractArray{T}) where {T, CT<:BFGSCache{T}}
    C.B = copy(B)
    C.S = copy(B)
    mul!(C.S, -o.η, C.S)
end