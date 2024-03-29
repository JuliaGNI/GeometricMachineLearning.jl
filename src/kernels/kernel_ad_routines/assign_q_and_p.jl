"""
This implements the custom pullback for assign_q_and_p 
"""

function ChainRulesCore.rrule(::typeof(assign_q_and_p), x::AbstractArray, N::Integer)
    qp = assign_q_and_p(x, N)
    function assign_q_and_p_pullback(qp_diff)
        f̄ = NoTangent()
        concat = @thunk vcat(qp_diff...)
        return f̄, concat, NoTangent()
    end
    return qp, assign_q_and_p_pullback
end