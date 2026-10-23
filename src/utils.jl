# overload norm
function _norm(dx::NamedTupleOfArrays{<:Any, (:q, :p)})
    T = eltype(dx.q)
    (norm(dx.q) + norm(dx.p)) / √(T(2))
end # we need this because of a Zygote problem
function _norm(dx::NamedTuple)
    n = sum(map(norm, dx))
    n / √(typeof(n)(length(dx)))
end
_norm(A::AbstractArray) = norm(A)

# overloaded - operation
function _diff(dx₁::NT, dx₂::NT) where {NT <: NamedTupleOfArrays{<:Any, (:q, :p)}}
    (q = dx₁.q - dx₂.q, p = dx₁.p - dx₂.p)
end # we need this because of a Zygote problem
_diff(dx₁::NamedTuple, dx₂::NamedTuple) = map(_diff, dx₁, dx₂)
_diff(A::AbstractArray, B::AbstractArray) = A - B
