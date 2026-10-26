# overload norm
function _norm(dx::NamedTuple)
    n = sum(map(norm, dx))
    n / √(typeof(n)(length(dx)))
end
_norm(A::AbstractArray) = norm(A)

# overloaded - operation
_diff(dx₁::NamedTuple, dx₂::NamedTuple) = map(_diff, dx₁, dx₂)
_diff(A::AbstractArray, B::AbstractArray) = A - B

# the `(q, p)` pair of arrays as one flat array, `q` first
_vcat(v::NamedTuple{(:q, :p), Tuple{AT, AT}}) where {AT <: AbstractArray} = vcat(v.q, v.p)
