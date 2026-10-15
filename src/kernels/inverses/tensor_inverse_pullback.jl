# The pullback of `tensor_inverse2` … `tensor_inverse5`, whose `rrule`s
# `scripts/utilities/inverse_generator.jl` writes: for the inverse `out`, slice by slice,
# `Ā = -out' * out_diff * out'`.
function _tensor_inverse_pullback(out::AbstractArray{T, 3}, out_diff) where {T}
    out_diff = _seed(T, out_diff)
    out_diff isa AbstractZero && return NoTangent(), ZeroTangent()

    NoTangent(),
    -tensor_transpose_tensor_mul(out, tensor_tensor_transpose_mul(out_diff, out))
end
