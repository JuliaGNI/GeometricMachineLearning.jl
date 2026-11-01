"""
    assign_q_and_p(x::AbstractArray, N::Integer)

Split `x` along its first axis, and return the `(q, p)` `NamedTuple` of the rows at the indices
`1:N` and at the indices `N + 1:2N`, as indexing `x` returns them: copies for an `Array`, and
ranges for a range. The indices are these whatever the first axis of `x` is, so an offset axis is
split at index `1`, not at its first row. Rows at other indices are ignored, and the other axes are
kept. For `N ≤ 0` both halves are empty.

Throws a `BoundsError` when the first axis of `x` does not hold the indices `1:2N`, and an
`ArgumentError` when `x` has no axis.

Slicing works on every backend, and Zygote differentiates it, so the split needs neither a kernel
nor an `rrule`.
"""
function assign_q_and_p(x::AbstractArray, N::Integer)
    rest = ntuple(Returns(:), ndims(x) - 1)
    (q = x[1:N, rest...], p = x[(N + 1):(2N), rest...])
end
