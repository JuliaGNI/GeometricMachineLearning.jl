@doc raw"""
This is an implementation of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer. 
"""
struct BFGSOptimizer{T<:Real} <: OptimizerMethod
    η::T
    δ::T

    function BFGSOptimizer(η::T = 1f-2, δ=1f-7) where T 
        new{T}(η, T(δ))
    end
end

@doc raw"""
Optimization for an entire neural networks with BFGS. What is different in this case is that we still have to initialize the cache.

If `o.step == 1`, then we initialize the cache
"""
function update!(o::Optimizer{<:BFGSOptimizer}, C::CT, B::AbstractArray{T}) where {T, CT<:BFGSCache{T}}
    if o.step == 1
        bfgs_initialization!(o, C, B)
    else
        bfgs_update!(o, C, B)   
    end
end

function bfgs_update!(o::Optimizer{<:BFGSOptimizer}, C::CT, B::AbstractArray{T}) where {T, CT<:BFGSCache{T}}
    # in the first step we compute the difference between the current and the previous mapped gradients:
    Y = vec(B - C.B) 
    # in the second step we update H (write a test to check that this preserves symmetry)
    vecS = vec(C.S)
    # the *term for the second condition* appears many times in the expression.
    SY = vecS' * Y + o.method.δ
    # C.H .= C.H + (SY + Y' * C.H * Y) / (SY ^ 2) * vecS * vecS' - (C.H * Y * vecS' + vecS * (C.H * Y)' ) / SY
    # the two computations of the H matrix should be equivalent. Check this!!
    HY = C.H * Y
    C.H .= C.H - HY * HY' / (Y' * HY + o.method.δ) + vecS * vecS' / SY
    # in the third step we compute the final velocity
    mul!(vecS, C.H, vec(B))
    mul!(C.S, -o.method.η, C.S)
    assign!(C.B, copy(B))
    assign!(B, copy(C.S))
end

function bfgs_initialization!(o::Optimizer{<:BFGSOptimizer}, C::CT, B::AbstractArray{T}) where {T, CT<:BFGSCache{T}}
    mul!(C.S, -o.method.η, B)
    assign!(C.B, copy(B))
    assign!(B, copy(C.S))
end