using GeometricMachineLearning
using LinearAlgebra: I, qr!
using Test
import Random

Random.seed!(123)

function test_setup(n2::Int, T::DataType)
    @assert iseven(n2)
    n = n2 ÷ 2

    one_mat = Matrix{T}(I(n))
    @test PoissonTensor(n2, T) ≈
          hcat(vcat(zero(one_mat), -one_mat), vcat(one_mat, zero(one_mat)))
end

@test Matrix(PoissonTensor(CPU(), 4, Float16)) ==
      Float16[0 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0]

function test_application(n2::Int, T::DataType)
    @assert iseven(n2)

    𝕁 = PoissonTensor(n2, T)
    x = rand(T, n2)
    y = rand(T, n2)

    @test 𝕁(x, y) ≈ x' * (𝕁 * y) ≈ -𝕁(y, x)
end

function test_application_nt(n2::Int, T::DataType)
    @assert iseven(n2)

    𝕁 = PoissonTensor(n2, T)
    n = n2 ÷ 2
    x = (q = rand(T, n), p = rand(T, n))
    y = (q = rand(T, n), p = rand(T, n))

    @test 𝕁(x, y) ≈ x.q' * y.p - x.p' * y.q ≈ -𝕁(y, x)
end

function test_application_to_nt(n2::Int, T::DataType)
    @assert iseven(n2)

    𝕁 = PoissonTensor(n2, T)
    n = n2 ÷ 2
    qp = (q = rand(T, n), p = rand(T, n))

    # Compared field by field rather than with `≈` on the two `NamedTuple`s: `Base` defines no `≈`
    # for a pair of `NamedTuple`s, and a method for it here would be piracy on `Base`.
    out = 𝕁 * qp
    @test out.q ≈ qp.p
    @test out.p ≈ -qp.q
end

for n2 in 2:2:10
    for T in (Float32, Float64)
        test_setup(n2, T)
        test_application(n2, T)
        test_application_nt(n2, T)
        test_application_to_nt(n2, T)
    end
end

# The three array methods of `*` take a `Strided…` right-hand side, so that they do not claim every
# `*(::AbstractMatrix, ::X)` another package defines for its own `X`. Two properties have to hold
# for that bound to be safe, and neither is visible in the ambiguity count.
@testset "a non-strided right-hand side falls through to the generic multiply" begin
    𝕁 = PoissonTensor(4, Float64)
    A = rand(4, 3)

    # A `SubArray` with vector indices is not strided, so it misses the specialised method. The
    # generic `AbstractMatrix` multiply has to give the same answer, because `PoissonTensor`
    # carries `getindex` and `size` and is the matrix it claims to be.
    nonstrided = view(A, [1, 2, 3, 4], :)
    @test !(nonstrided isa StridedMatrix)
    @test 𝕁 * nonstrided ≈ 𝕁 * A

    # And a strided `view` still reaches the specialised method, so the fast path is not lost for
    # the arguments this package actually builds.
    strided = view(A, :, 1:2)
    @test strided isa StridedMatrix
    @test 𝕁 * strided ≈ 𝕁 * A[:, 1:2]
end

@testset "all four right-hand sides of `*` give `(q; p) -> (p; -q)`" begin
    # One assertion per method, because the three array methods take a `Strided…` argument each and
    # the `(q, p)` one takes a `NamedTuple`. The value is what must not move: a narrowed signature
    # is only safe if every shape still answers the same.
    T = Float32
    n2, n = 4, 2
    𝕁 = PoissonTensor(n2, T)

    v = rand(T, n2)
    @test 𝕁 * v == vcat(v[(n + 1):n2], -v[1:n])

    m = rand(T, n2, 3)
    @test 𝕁 * m == vcat(m[(n + 1):n2, :], -m[1:n, :])

    t = rand(T, n2, 3, 2)
    @test 𝕁 * t == vcat(t[(n + 1):n2, :, :], -t[1:n, :, :])

    qp = (q = rand(T, n), p = rand(T, n))
    @test (𝕁 * qp).q == qp.p
    @test (𝕁 * qp).p == -qp.q

    # And each array result agrees with the plain matrix product, which is what a non-strided
    # argument falls through to.
    @test 𝕁 * v == Matrix(𝕁) * v
    @test 𝕁 * m == Matrix(𝕁) * m
end

@testset "`getindex` is typed on `Int` and every other index shape still works" begin
    # The scalar method is the only one defined; a range, a `Colon` and a `CartesianIndex` are
    # `Base`'s generic `AbstractArray` indexing built on top of it, and each has to give what the
    # wrapped matrix gives for the same index pair.
    𝕁 = PoissonTensor(4, Float64)
    M = Matrix(𝕁)

    @test 𝕁[1, 3] == M[1, 3]
    @test 𝕁[3, 1] == M[3, 1]
    @test 𝕁[1:2, :] == M[1:2, :]
    @test 𝕁[:, 1] == M[:, 1]
    @test 𝕁[CartesianIndex(3, 1)] == M[3, 1]
    @test 𝕁[end, end] == M[end, end]
end

@testset "the narrowed `*` methods still resolve against an upstream special type" begin
    # The witness for the ambiguity the `Strided…` bound closes: both names are exported by this
    # package, so a caller can write this call, and it has to resolve to GeometricOptimizers' own
    # left-multiply rather than being ambiguous against this package's `*`.
    𝕁 = PoissonTensor(4, Float32)
    Y = StiefelManifold(Matrix(qr!(rand(Float32, 4, 2)).Q)[:, 1:2])

    @test 𝕁 * Y ≈ Matrix(𝕁) * Y.A
end

# `PoissonTensor(n2)` (no backend, no type) keeps its documented `Float64` default.
@test eltype(PoissonTensor(4)) == Float64

# `PoissonTensor(backend::Backend, n2::Int)` has no method: a caller that names a backend must
# also name an element type. See the "Removed (breaking)" section of `CHANGELOG.md`.
@test_throws MethodError PoissonTensor(CPU(), 4)
