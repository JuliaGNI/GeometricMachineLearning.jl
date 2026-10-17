@doc raw"""
    tensor_cayley(C::AbstractArray{T, 3})

Compute the Cayley transform of every slice of the three-tensor `C`: the slice ``k`` of the result
is

```math
    \Lambda_k = (\mathbb{I} + C_k)^{-1}(\mathbb{I} - C_k),
```

computed by row-pivoted Gauss–Jordan elimination on the pair ``(\mathbb{I} + C_k,
\mathbb{I} - C_k)``, which leaves ``\Lambda_k`` in the second member. This is the transform of
[`VolumePreservingAttention`](@ref), which forms ``C`` skew-symmetric.

`C` is assumed skew-symmetric, and this is not checked: a check would cost a reduction per slice on
a device. For nonsingular ``\mathbb{I} + C_k``, the algorithm computes the Cayley transform.

The tests exercise sizes 2 to 8 at scales 1 and 1e2. On exactly skew inputs, they bound each
slice's orthogonality error ``\|\Lambda_k^T\Lambda_k - \mathbb{I}\|_\infty`` by
``10n\varepsilon(1 + \|C_k\|_2^2)``, where `n = size(C, 1)` and ``\varepsilon`` is the element
type's machine epsilon. They also bound the batch maximum error by four times the batch maximum
of a dense `(I + C_k) \ (I - C_k)` reference, including the layer's inputs that are skew only to
rounding. These are tested bounds, not guarantees for arbitrary magnitudes: elimination can
overflow at large scales. For skew ``C_k``, the symmetric part of ``\mathbb{I} + C_k`` is
positive definite.

The element type `T` is an `AbstractFloat`. Every buffer comes from `similar(C)`, so the result has
the element type `T` and lives on the backend of `C`. The loop bound is read from `size(C, 1)` at
run time, so a slice of any size is solved by the same kernel.

A slice whose ``\mathbb{I} + C_k`` is singular gives non-finite entries in the slice of that `k` and
no error, as the closed forms do, which divide by the determinant of the slice; a `NaN` or an `Inf`
in `C` spreads within its own slice. A slice that is not square raises a `DimensionMismatch`, and
an empty batch is not reached by a caller of this function.
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
@kernel function tensor_cayley_kernel!(A, Λ, C)
    k = @index(Global)
    n = size(C, 1)

    for i in 1:n, j in 1:n

        δ = i == j ? one(eltype(C)) : zero(eltype(C))
        A[i, j, k] = δ + C[i, j, k]
        Λ[i, j, k] = δ - C[i, j, k]
    end

    for col in 1:n
        pivot = col
        largest = abs(A[col, col, k])
        for r in (col + 1):n
            candidate = abs(A[r, col, k])
            if candidate > largest
                largest = candidate
                pivot = r
            end
        end

        if pivot != col
            for X in (A, Λ), j in 1:n

                a = X[col, j, k]
                X[col, j, k] = X[pivot, j, k]
                X[pivot, j, k] = a
            end
        end

        diagonal = A[col, col, k]
        for j in 1:n
            A[col, j, k] = A[col, j, k] / diagonal
            Λ[col, j, k] = Λ[col, j, k] / diagonal
        end

        for r in 1:n
            factor = A[r, col, k]
            if r != col && factor != zero(eltype(C))
                for j in 1:n
                    A[r, j, k] = A[r, j, k] - factor * A[col, j, k]
                    Λ[r, j, k] = Λ[r, j, k] - factor * Λ[col, j, k]
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
        Y = Λ̄ + tensor_transpose_tensor_mul(Λ, Λ̄)
        Z = tensor_tensor_transpose_mul(Y, Λ)
        NoTangent(), @. -(Y + Z) / T(2)
    end

    Λ, tensor_cayley_pullback
end
