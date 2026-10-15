@doc raw"""
    _mul(W::AbstractMatrix, x::AbstractArray)
    _mul(x::AbstractArray{<:Number, 3}, W::AbstractMatrix)
    _mul(x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3})

Multiply a matrix and a tensor, or two tensors, slice by slice along the third axis.

- The **left product** multiplies `W` onto `x` from the left. For a vector or a matrix `x` it is
  `W * x`. For a tensor it is one GEMM: `x` is reshaped into a matrix with `size(x, 1)` rows, so
  that `_mul(W, x)[:, :, k] == W * x[:, :, k]`.
- The **right product** multiplies `W` onto every slice of the tensor `x` from the right,
  `_mul(x, W)[:, :, k] == x[:, :, k] * W`. It is one GEMM between two `permutedims`.
- The **tensor product** multiplies the slices of two tensors pairwise,
  `_mul(x, y)[:, :, k] == x[:, :, k] * y[:, :, k]`, with the `tensor_tensor_mul` kernel.

A `SymmetricMatrix`, `SkewSymMatrix`, `StrictlyLowerTriangular` or `StrictlyUpperTriangular` weight
multiplies a vector or a matrix through its own `*`. Before a GEMM with a tensor, and as the right
argument of a left product, it is made dense once per call, on its own backend, and its gradient
reaches the storage through `ProjectTo`. Element types are promoted as `*` promotes them.

# Examples

A left product:

```jldoctest
using GeometricMachineLearning: _mul

B = [1 1 1; 1 1 1; 1 1 1;;; 2 2 2; 2 2 2; 2 2 2]
A = [3 0 0; 0 2 0; 0 0 1]

_mul(A, B)

# output

3×3×2 Array{Int64, 3}:
[:, :, 1] =
 3  3  3
 2  2  2
 1  1  1

[:, :, 2] =
 6  6  6
 4  4  4
 2  2  2
```

A right product:

```jldoctest
using GeometricMachineLearning: _mul

A = [1 1 1; 1 1 1; 1 1 1;;; 2 2 2; 2 2 2; 2 2 2]
B = [3 0 0; 0 2 0; 0 0 1]

_mul(A, B)

# output

3×3×2 Array{Int64, 3}:
[:, :, 1] =
 3  2  1
 3  2  1
 3  2  1

[:, :, 2] =
 6  4  2
 6  4  2
 6  4  2
```
"""
_mul(W::AbstractMatrix, x::AbstractVecOrMat) = W * _dense(x)

function _mul(W::AbstractMatrix, x::AbstractArray)
    reshape(_dense(W) * _matrix(x), size(W, 1), Base.tail(size(x))...)
end

function _mul(x::AbstractArray{<:Number, 3}, W::AbstractMatrix)
    xW = reshape(permutedims(x, (1, 3, 2)), :, size(x, 2)) * _dense(W)
    permutedims(reshape(xW, size(x, 1), size(x, 3), size(W, 2)), (1, 3, 2))
end

_mul(x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3}) = tensor_tensor_mul(x, y)

# A structured matrix as a dense matrix on its own backend; `one` allocates the identity there.
function _dense(W::Union{
        SymmetricMatrix, SkewSymMatrix, StrictlyLowerTriangular, StrictlyUpperTriangular})
    W * one(W)
end
_dense(x::AbstractArray) = x

# `x` as a dense matrix: its trailing axes folded into the second, or, for a structured matrix,
# `_dense(x)`. A matrix is not reshaped, which would hide an `Adjoint` behind a `ReshapedArray`.
_matrix(x::AbstractMatrix) = _dense(x)
_matrix(x::AbstractArray) = reshape(x, size(x, 1), :)

# Without this rule a `Matrix` parameter `P` multiplied as `_mul(P', x)` gets an `Adjoint`
# gradient: the generic rule of `*` gives `W` a dense cotangent `C`, and the rule of `adjoint` hands
# `C'` to `P`. This rule gives `W` the cotangent `C'` instead, so that `P` receives `C`.
function ChainRulesCore.rrule(::typeof(_mul), W::Adjoint{<:Number, <:AbstractMatrix}, x::AbstractArray)
    y = _mul(W, x)
    project_x = ProjectTo(x)
    function _mul_adjoint_pullback(ȳ)
        ȳ = _seed(eltype(y), ȳ)
        ȳ isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        ȳ_matrix = _matrix(ȳ)
        x_matrix = _matrix(x)
        W̄ = convert(AbstractMatrix{eltype(W)}, x_matrix * ȳ_matrix')'
        x̄ = project_x(reshape(W' * ȳ_matrix, size(x)))
        NoTangent(), W̄, x̄
    end
    y, _mul_adjoint_pullback
end
