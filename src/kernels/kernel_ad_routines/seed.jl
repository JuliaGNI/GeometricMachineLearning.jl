# The seed `ȳ` of a kernel rule's pullback, for a primal of element type `T`: unthunked, and
# converted to `T`, because the kernels take one element type for all their arrays. An
# `AbstractZero` passes through, and every rule answers it with zeros before a kernel runs.
_seed(::Type{T}, ȳ) where {T} = _seed_eltype(T, unthunk(ȳ))
_seed_eltype(::Type, ȳ::AbstractZero) = ȳ
_seed_eltype(::Type{T}, ȳ::AbstractArray) where {T} = convert(AbstractArray{T}, ȳ)
