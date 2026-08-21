"""
This implements the custom pullback for assign_q_and_p 
"""

# `assign_q_and_p` returns a `(q, p)` `NamedTuple`, so its cotangent is one too -- and either the
# tangent itself or its two components may arrive as a thunk. Splatting it straight into `vcat`
# without unthunking concatenates nothing: it builds a two-element `Vector{Thunk}`, which is a
# silently wrong gradient rather than an error.
_vcat_qp(qp̄) = vcat(unthunk(qp̄.q), unthunk(qp̄.p))

function ChainRulesCore.rrule(::typeof(assign_q_and_p), x::AbstractArray, N::Integer)
    qp = assign_q_and_p(x, N)
    function assign_q_and_p_pullback(qp_diff)
        f̄ = NoTangent()
        concat = @thunk _vcat_qp(unthunk(qp_diff))
        return f̄, concat, NoTangent()
    end
    return qp, assign_q_and_p_pullback
end
