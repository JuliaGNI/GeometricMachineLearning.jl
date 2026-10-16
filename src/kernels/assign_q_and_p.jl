# Split `x` along its first axis into the `(q, p)` `NamedTuple` of its first `N` rows and the `N`
# rows after them. Rows from `2N + 1` on are ignored. Slicing works on every backend, and Zygote
# differentiates it, so the split needs neither a kernel nor an `rrule`.
function assign_q_and_p(x::AbstractArray, N::Integer)
    rest = ntuple(Returns(:), ndims(x) - 1)
    (q = x[1:N, rest...], p = x[(N + 1):(2N), rest...])
end
