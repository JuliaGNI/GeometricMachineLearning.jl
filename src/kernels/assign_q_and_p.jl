# Split `x` along its first axis into the `(q, p)` `NamedTuple` of its first `N` rows and the `N`
# rows after them. Rows from `2N + 1` on are ignored. Slicing works on every backend, and Zygote
# differentiates it, so the split needs neither a kernel nor an `rrule`.
assign_q_and_p(x::AbstractVector, N::Integer) = (q = x[1:N], p = x[(N + 1):(2N)])
assign_q_and_p(x::AbstractMatrix, N::Integer) = (q = x[1:N, :], p = x[(N + 1):(2N), :])
function assign_q_and_p(x::AbstractArray{<:Number, 3}, N::Integer)
    (q = x[1:N, :, :], p = x[(N + 1):(2N), :, :])
end
