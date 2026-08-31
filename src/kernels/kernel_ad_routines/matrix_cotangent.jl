# `_matrix_cotangent(B, dB)` gives the cotangent `dB` of a matrix argument `B` the shape *and* the
# array type of `B`. It is a plain comment rather than a docstring on purpose: `docs/make.jl` runs
# Documenter with the default `checkdocs = :all`, so every docstring in the module has to be placed
# in the manual.
#
# Two things need undoing. The kernel rules build a matrix argument's cotangent by contracting the
# third axis with `sum(_, dims = 3)`, which leaves a trailing singleton axis; a cotangent has to
# have the shape of the primal it belongs to, so that axis comes off.
#
# The array type matters for `Adjoint`, which is what `B` is wherever a layer multiplies by a
# transposed weight -- `MultiHeadAttention` computes `mat_tensor_mul(ps.PQ[key]', x)`. ChainRules'
# `rrule` for `adjoint` unwraps an `Adjoint` cotangent to its dense parent but leaves a dense one
# wrapped, so returning a plain `Matrix` here hands the caller an `Adjoint{T, Matrix{T}}` as the
# gradient of a `Matrix{T}` parameter. `ProjectTo` does not help: for a real `Adjoint` it is a
# `ProjectTo{AbstractArray}`, which keeps whatever wrapper the tangent arrived with.
_matrix_cotangent(::AbstractMatrix, dB::AbstractArray{<:Number, 3}) = dropdims(dB; dims = 3)
function _matrix_cotangent(::Adjoint, dB::AbstractArray{<:Number, 3})
    collect(dropdims(dB; dims = 3)')'
end
