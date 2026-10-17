@doc raw"""
    tensor_cayley(C::AbstractArray{T, 3})

Compute the Cayley transform of every slice of the three-tensor `C`: the slice ``k`` of the result
is

```math
    \Lambda_k = (\mathbb{I} + C_k)^{-1}(\mathbb{I} - C_k),
```

computed by row-pivoted Gauss–Jordan elimination on the pair ``(\mathbb{I} + C_k,
\mathbb{I} - C_k)``, one work item per slice, for slices of any size. This is the transform of
[`VolumePreservingAttention`](@ref), which forms ``C`` skew-symmetric. `C` is assumed skew and this
is not checked; for skew ``C_k``, ``\mathbb{I} + C_k`` is nonsingular.

The result is `similar(C)`: it has the element type of `C` and lives on its backend. A singular
``\mathbb{I} + C_k``, or a `NaN` or an `Inf` in ``C_k``, gives non-finite entries in slice ``k``
and no error. A slice that is not square raises a `DimensionMismatch`.
"""
function tensor_cayley(C::AbstractArray{T, 3}) where {T}
    n = size(C, 1)
    size(C, 2) == n ||
        throw(DimensionMismatch("a slice of C is $(n)×$(size(C, 2)); the slices must be square"))

    backend = networkbackend(C)
    Λ = similar(C)
    A = similar(C)

    cayley! = tensor_cayley_kernel!(backend)
    cayley!(A, Λ, C, ndrange = size(C, 3))

    Λ
end

# `(I + Cₖ) Λₖ = (I - Cₖ)` for every slice at once, one work item per slice: `A` is filled with
# `I + C`, `Λ` with `I - C`, and the row operations that turn `A` into `I` are also applied to `Λ`.
# The pivot of a column is the row of the largest magnitude in it, and the swap moves both members:
# the two are one augmented system, and a swap on one of them alone solves a permuted system.
# Every index is in bounds: `tensor_cayley` checks that the slices are square, `A` and `Λ` are
# `similar(C)`, and KernelAbstractions masks `k` to the `ndrange`.
@kernel function tensor_cayley_kernel!(A, Λ, C)
    k = @index(Global)
    n = size(C, 1)

    @inbounds begin
        for j in 1:n, i in 1:n

            δ = ifelse(i == j, one(eltype(C)), zero(eltype(C)))
            A[i, j, k] = δ + C[i, j, k]
            Λ[i, j, k] = δ - C[i, j, k]
        end

        for col in 1:n
            pivot = col
            for r in (col + 1):n
                abs(A[r, col, k]) > abs(A[pivot, col, k]) && (pivot = r)
            end

            for j in 1:n
                A[col, j, k], A[pivot, j, k] = A[pivot, j, k], A[col, j, k]
                Λ[col, j, k], Λ[pivot, j, k] = Λ[pivot, j, k], Λ[col, j, k]
            end

            d = A[col, col, k]
            for j in 1:n
                A[col, j, k] /= d
                Λ[col, j, k] /= d
            end

            for r in 1:n
                r == col && continue
                f = A[r, col, k]
                for j in 1:n
                    A[r, j, k] -= f * A[col, j, k]
                    Λ[r, j, k] -= f * Λ[col, j, k]
                end
            end
        end
    end

    nothing
end

# `Λ = (I + C)⁻¹(I - C)` gives `(I + Λ) = 2(I + C)⁻¹`, so the cotangent of `C` under the Frobenius
# product is `C̄ = -(I + Λ)ᵀ Λ̄ (I + Λ)ᵀ / 2`, from `Λ` alone: no solve, and no identity tensor.
function ChainRulesCore.rrule(::typeof(tensor_cayley), C::AbstractArray{T, 3}) where {T}
    Λ = tensor_cayley(C)

    function tensor_cayley_pullback(Λ̄)
        Λ̄ = _seed(T, Λ̄)
        Λ̄ isa AbstractZero && return NoTangent(), ZeroTangent()
        Y = tensor_transpose_tensor_mul(Λ, Λ̄)
        Y .+= Λ̄
        C̄ = tensor_tensor_transpose_mul(Y, Λ)
        @. C̄ = -(Y + C̄) / T(2)
        NoTangent(), C̄
    end

    Λ, tensor_cayley_pullback
end
