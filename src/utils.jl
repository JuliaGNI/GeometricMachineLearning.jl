# overload norm
function _norm(dx::NamedTuple)
    n = sum(map(norm, dx))
    n / √(typeof(n)(length(dx)))
end
_norm(A::AbstractArray) = norm(A)

# overloaded - operation
_diff(dx₁::NamedTuple, dx₂::NamedTuple) = map(_diff, dx₁, dx₂)
_diff(A::AbstractArray, B::AbstractArray) = A - B
