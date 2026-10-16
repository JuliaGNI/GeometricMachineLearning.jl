# `tensor_cayley`: that its rows are skew on every input the layer sends it, that its orthogonality
# error is the error of a row-pivoted elimination, that its value is the value of the transform, and
# that its pullback is the derivative of that value.
#
# The three input sets are the ones the layer produces: a hand-built skew tensor, the `:arbitrary`
# path and the `:skew_sym` path, both of them through the layer's own functions. The reference is the
# per-slice solve `(I + Cₖ) \ (I - Cₖ)` in the working precision, written here, because the closed
# forms are what `tensor_cayley` replaces.

using GeometricMachineLearning
using GeometricMachineLearning: tensor_cayley, tensor_mat_skew_sym_assign,
                                tensor_transpose_tensor_mul, _mul
using ChainRulesCore: AbstractZero, ZeroTangent, rrule
using LinearAlgebra: BLAS, I, opnorm
using Test
import Random

Random.seed!(1234)

# The references below are solves of 8×8 systems, thousands of them, and threaded LAPACK spends more
# time waking its threads than solving: with BLAS on this machine's twelve threads the orthogonality
# rows over their 84 batches take about two minutes, and 2.6 s on one thread, where the value rows
# take 10.9 s. The file pins one thread for its own work and restores the count it found at the end,
# so no later testset inherits it.
const BLAS_THREADS = BLAS.get_num_threads()
BLAS.set_num_threads(1)

const SLICES = 1000
const SCALES = (1, 1e2)

function identity_tensor(::Type{T}, n::Int, k::Int) where {T}
    E = zeros(T, n, n, k)
    for i in 1:n
        E[i, i, :] .= one(T)
    end
    E
end

# The largest `‖Cₖ‖₂` over the slices of `C`, as a `Float64`.
function largest_norm(C::AbstractArray)
    maximum(Float64(opnorm(@view(C[:, :, k]), 2)) for k in axes(C, 3))
end

# The tensor scaled so that its largest `‖Cₖ‖₂` is `s`.
scale_to(C::AbstractArray, s) = C * (s / largest_norm(C))

# (i) A hand-built skew tensor, `R - Rᵀ` for `R = s * randn`, which is skew to rounding.
function skew_tensor(::Type{T}, n::Int, s, k::Int = SLICES) where {T}
    R = T(s) * randn(T, n, n, k)
    R - permutedims(R, (2, 1, 3))
end

# (ii) The `:arbitrary` path of the layer: `tensor_mat_skew_sym_assign(x, A) / √M`, then
# `x_interim - permutedims(x_interim, (2, 1, 3))`. That path is linear in `A`, so the whole tensor is
# scaled by one factor on `A` to put its largest `‖Cₖ‖₂` at `s`.
function arbitrary_tensor(::Type{T}, n::Int, s) where {T}
    x = randn(T, n, n, SLICES)
    A = randn(T, n, n)
    scale_to(arbitrary_path(x, A), s)
end

function arbitrary_path(x::AbstractArray{T, 3}, A::AbstractMatrix) where {T}
    x_interim = tensor_mat_skew_sym_assign(x, A) / T(√size(x, 1))
    x_interim - permutedims(x_interim, (2, 1, 3))
end

# (iii) The `:skew_sym` path of the layer: `Zᵀ A Z` for a `SkewSymMatrix` `A`. That is quadratic in
# `Z`, so `Z` is scaled by the square root of the factor that puts the largest `‖Cₖ‖₂` at `s`.
function skew_sym_tensor(::Type{T}, n::Int, s) where {T}
    x = randn(T, n, n, SLICES)
    A = rand(SkewSymMatrix{T}, n)
    C = skew_sym_path(x, A)
    skew_sym_path(x * T(sqrt(s / largest_norm(C))), A)
end

function skew_sym_path(x::AbstractArray{T, 3}, A::AbstractMatrix) where {T}
    tensor_transpose_tensor_mul(x, _mul(A, x))
end

# The per-slice reference of the orthogonality rows and of the value row: the solve
# `(I + Cₖ) \ (I - Cₖ)` in the working precision, and in `BigFloat` for the value row. The two
# agree to the last bits of `T`, so the value row's reference is the transform itself rather than
# another elimination.
function reference_solve(::Type{T}, Ck::AbstractMatrix) where {T}
    Id = Matrix{T}(I, size(Ck))
    (Id + Ck) \ (Id - Ck)
end

function big_reference_solve(::Type{T}, Ck::AbstractMatrix) where {T}
    Id = Matrix{BigFloat}(I, size(Ck))
    C = BigFloat.(Ck)
    T.((Id + C) \ (Id - C))
end

# The same maximum over the pivoted references, which is what the ratio clause compares against.
function reference_orthogonality_error(C::AbstractArray{T, 3}) where {T}
    n = size(C, 1)
    E = Matrix{T}(I, n, n)
    worst = 0.0
    for k in axes(C, 3)
        X = reference_solve(T, @view(C[:, :, k]))
        worst = max(worst, Float64(maximum(abs, X' * X - E)))
    end
    worst
end

# The bounds of the orthogonality rows, per slice: the absolute clause `10 n eps(T) (1 + ‖Cₖ‖₂²)`,
# and the ratio clause's own factor on the reference's error.
function bound_per_slice(::Type{T}, Ck::AbstractMatrix, n::Int) where {T}
    10.0 * n * eps(T) * (1.0 + Float64(opnorm(Ck, 2))^2)
end

# Every row of the orthogonality check: the ratio against the pivoted reference on all three input
# sets, and the absolute bound on the two exactly skew ones.
function orthogonality_rows(::Type{T}, n::Int, s) where {T}
    kernel_error = Ref(0.0)
    reference_error = Ref(0.0)
    worst_absolute = Ref(0.0)
    for (set, C) in (("hand-built", skew_tensor(T, n, s)),
        ("arbitrary", arbitrary_tensor(T, n, s)),
        ("skew_sym", skew_sym_tensor(T, n, s)))
        Λ = tensor_cayley(C)
        E = identity_tensor(T, n, size(C, 3))
        ΛᵀΛ = tensor_transpose_tensor_mul(Λ, Λ)
        kernel_error[] = Float64(maximum(abs, ΛᵀΛ - E))
        reference_error[] = reference_orthogonality_error(C)

        # The ratio clause holds on every input set; the reference's own error is not zero, so the
        # factor is a real bound rather than an equality.
        @test kernel_error[] <= 4 * reference_error[]

        if set != "skew_sym"
            for k in axes(C, 3)
                bound = bound_per_slice(T, @view(C[:, :, k]), n)
                error_k = Float64(maximum(abs, (@view ΛᵀΛ[:, :, k]) .- (@view E[:, :, k])))
                worst_absolute[] = max(worst_absolute[], error_k / bound)
            end
            @test worst_absolute[] <= 1
        end
    end
end

# The value row: per slice, the distance from the `BigFloat` solve, rounded to `T`, under the same
# bound. `Λᵀ` is orthogonal wherever `Λ` is, so the orthogonality rows alone do not pin the value.
function value_rows(::Type{T}, n::Int, s) where {T}
    for (set, C) in (("hand-built", skew_tensor(T, n, s)),
        ("arbitrary", arbitrary_tensor(T, n, s)))
        Λ = tensor_cayley(C)
        worst = Ref(0.0)
        for k in axes(C, 3)
            bound = bound_per_slice(T, @view(C[:, :, k]), n)
            error_k = Float64(maximum(abs, (@view Λ[:, :, k]) .-
                                           big_reference_solve(T, @view(C[:, :, k]))))
            worst[] = max(worst[], error_k / bound)
        end
        @test worst[] <= 1
    end
end

@testset "tensor_cayley: the input of the layer is always skew" begin
    for T in (Float32, Float64), n in 2:4

        x = randn(T, n, n, 7)
        A = randn(T, n, n)
        # The `:arbitrary` path is exactly skew: one subtraction of a permutation of the same array.
        C = arbitrary_path(x, A)
        @test iszero(C + permutedims(C, (2, 1, 3)))
        # The `:skew_sym` path is skew only to rounding, so nothing is asserted exactly on it. Its
        # deviation is far below the orthogonality rows' bound, which is what those rows measure.
        C2 = skew_sym_path(x, rand(SkewSymMatrix{T}, n))
        deviation = maximum(abs, C2 + permutedims(C2, (2, 1, 3)))
        @test deviation <= 100 * eps(T) * n * maximum(abs, C2)
    end
end

@testset "tensor_cayley: the orthogonality error and the value, N = $n, s = $s" for T in (Float32, Float64),
    s in SCALES, n in 2:8

    orthogonality_rows(T, n, s)
    value_rows(T, n, s)
end

@testset "tensor_cayley: the pullback, N = $n" for T in (Float32, Float64),
    n in (2, 3, 5, 8)

    C = skew_tensor(T, n, 1, 7)
    Λ̄ = randn(T, n, n, 7)

    if T === Float64
        # The relative error against a central difference of `⟨Λ̄, tensor_cayley(·)⟩` along a random
        # direction, with `h = cbrt(eps)`.
        C̄ = rrule(tensor_cayley, C)[2](Λ̄)[2]
        D = randn(T, size(C))
        h = cbrt(eps(T))
        loss(X) = sum(Λ̄ .* tensor_cayley(X))
        fd = (loss(C + h * D) - loss(C - h * D)) / (2h)
        @test abs(sum(C̄ .* D) - fd) / abs(fd) <= 1e-6
    else
        # The `Float32` pullback against the `Float64` one, under the bound of the orthogonality
        # rows: the cotangent is compared relatively, because its scale is the data's.
        C̄32 = rrule(tensor_cayley, C)[2](Λ̄)[2]
        C̄64 = rrule(tensor_cayley, Float64.(C))[2](Float64.(Λ̄))[2]
        bound = 10 * n * eps(T) * (1 + largest_norm(C)^2)
        @test maximum(abs, C̄32 - Float32.(C̄64)) <= bound * maximum(abs, C̄64)
    end

    # The seed handling of every kernel rule: a zero seed, and a seed of the other precision.
    @test all(c -> c isa AbstractZero, rrule(tensor_cayley, C)[2](ZeroTangent()))
    @test eltype(rrule(tensor_cayley, C)[2](Float64.(Λ̄))[2]) === T
end

@testset "tensor_cayley: the result's element type, backend and shape" begin
    for T in (Float32, Float64), n in 2:8

        C = skew_tensor(T, n, 1)
        Λ = @inferred tensor_cayley(C)
        @test Λ isa Array{T, 3}
        @test size(Λ) == size(C)
        @test @inferred(tensor_cayley(C))[1, 1, 1] isa T
    end
    for T in (Float32, Float64)
        @test_throws DimensionMismatch tensor_cayley(rand(T, 3, 4, 2))
        @test_throws DimensionMismatch tensor_cayley(rand(T, 4, 3, 2))
    end
end

BLAS.set_num_threads(BLAS_THREADS)
