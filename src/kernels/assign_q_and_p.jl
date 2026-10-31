"""
    assign_q_and_p(x::AbstractArray, N::Integer)

Split `x` along its first axis, and return the `(q, p)` `NamedTuple` of copies of its first `N` rows
and of the `N` rows after them. Rows from `2N + 1` on are ignored, and the other axes are kept.

Throws a `BoundsError` when `x` has fewer than `2N` rows, and an `ArgumentError` when `x` has no
axis.

Slicing works on every backend, and Zygote differentiates it, so the split needs neither a kernel
nor an `rrule`. `HamiltonianNeuralNetworks` calls this function by its qualified name, so it is
public.
"""
function assign_q_and_p(x::AbstractArray, N::Integer)
    rest = ntuple(Returns(:), ndims(x) - 1)
    (q = x[1:N, rest...], p = x[(N + 1):(2N), rest...])
end
