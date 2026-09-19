
@doc raw"""
    PoissonTensor(n2)
    PoissonTensor(n2, T)
    PoissonTensor(backend, n2, T)

Returns a (canonical) Poisson tensor of size ``2n\times2n``:

```math
\mathbb{J}_{2n} = \begin{pmatrix}
\mathbb{O} & \mathbb{I}_n \\
-\mathbb{I}_n & \mathbb{O} \\
\end{pmatrix}
```

# Arguments

The element type `T` defaults to `Float64` and the backend to `CPU()`. A call that names a
`backend` has to name a `T` as well:
```jldoctest
using GeometricMachineLearning

backend = CPU()
T = Float16

PoissonTensor(backend, 4, T)

# output

4×4 PoissonTensor{Float16, Matrix{Float16}}:
  0.0   0.0  1.0  0.0
  0.0   0.0  0.0  1.0
 -1.0   0.0  0.0  0.0
  0.0  -1.0  0.0  0.0
```
"""
struct PoissonTensor{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    J::AT
    n::Int
end

# `Int`, not an untyped index pair: scalar `getindex` is the one method an `AbstractArray` has to
# supply, and everything else -- a range, a `Colon`, a `CartesianIndex` -- is built on it by `Base`.
# Left untyped, this method claims every index type, and it then collides with the `getindex`
# methods BandedMatrices and BlockArrays add to `AbstractMatrix` for `Block`, `BlockIndex` and
# `BandRangeType`: nine ambiguities, on the same type and of the same kind as the seventeen that
# the `Strided…` bound on `*` below removes. Neither package loads with this one alone, which is
# why the pile is only visible from inside the test suite.
Base.getindex(𝕁::PoissonTensor, i::Int, j::Int) = getindex(𝕁.J, i, j)

Base.size(𝕁::PoissonTensor) = size(𝕁.J)

function PoissonTensor(backend::Backend, n2::Int, T::DataType)
    @assert iseven(n2)
    n = n2÷2
    J = KernelAbstractions.zeros(backend, T, 2*n, 2*n)
    assign_ones_for_poisson_tensor! = assign_ones_for_poisson_tensor_kernel!(backend)
    assign_ones_for_poisson_tensor!(J, n, ndrange = n2)

    PoissonTensor{T, typeof(J)}(J, n)
end

PoissonTensor(n2::Int, T::DataType) = PoissonTensor(CPU(), n2, T)

PoissonTensor(n2::Int) = PoissonTensor(n2, Float64)

# A `PoissonTensor` constructor that takes a `backend` requires an element type argument too --
# there is no `PoissonTensor(backend, n2)` method. `Float64` and `Float32` disagree on Metal (Apple
# GPUs have no `Float64` at all), so a shared CPU/GPU default could only be wrong for one side; the
# caller states the type instead. See the "Removed (breaking)" section of `CHANGELOG.md`.

@kernel function assign_ones_for_poisson_tensor_kernel!(J::AbstractMatrix{T}, n::Int) where {T}
    i = @index(Global)
    J[map_index_for_poisson_tensor(i, n)...] = i ≤ n ? one(T) : -one(T)
end

function _vcat(v::NamedTuple{(:q, :p), Tuple{AT, AT}}) where {AT <: AbstractArray}
    vcat(v.q, v.p)
end

# The right-hand side of the three array methods is `Strided…`, not `Abstract…`, and the bound is
# load-bearing rather than cosmetic. `PoissonTensor <: AbstractMatrix{T}`, so a method claiming the
# whole of `AbstractVecOrMat` on the right collides with every `*(::AbstractMatrix, ::X)` that
# ArrayLayouts, FillArrays, Symbolics and GeometricOptimizers define for their own special array
# type `X` -- seventeen ambiguities, each of which throws a `MethodError` on a call a user can
# write, `PoissonTensor(4, Float32) * StiefelManifold(...)` among them. `StridedArray` excludes
# every such `X` while still covering what this package produces and consumes: `Array`, a strided
# `SubArray`, and the GPU arrays, because `CuArray` and `MtlArray` are `DenseArray`s. An `Adjoint`
# of a `Matrix` is not strided and takes the generic path below.
#
# A non-strided matrix or vector reaches Julia's generic `AbstractMatrix` multiply, which gives the
# same answer: this type carries `getindex` and `size` and *is* the matrix it claims
# to be, and `𝕁 * (q; p) = (p; -q)` either way. Only the fast path is given up, and only for an
# argument no call site here builds. There is no generic fallback for a 3-tensor right-hand side,
# so that method's narrowing turns a non-strided 3-tensor into a `MethodError`; nothing in the
# package, the tests or the scripts passes one.
Base.:*(::PoissonTensor, v::QPT) = (q = v.p, p = -v.q)
function Base.:*(𝕁::PoissonTensor{T}, v::StridedArray{T, 3}) where {T}
    _vcat(𝕁(assign_q_and_p(v, 𝕁.n)))
end
function Base.:*(𝕁::PoissonTensor{T}, v::StridedVector{T}) where {T}
    _vcat(𝕁(assign_q_and_p(v, 𝕁.n)))
end
function Base.:*(𝕁::PoissonTensor{T}, v::StridedMatrix{T}) where {T}
    _vcat(𝕁(assign_q_and_p(v, 𝕁.n)))
end

function (𝕁::PoissonTensor{T})(v₁::NT,
        v₂::NT) where {
        T, AT <: AbstractVector{T}, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    v₁.q' * v₂.p - v₁.p' * v₂.q
end

function (𝕁::PoissonTensor{T})(v₁::AbstractVector{T}, v₂::AbstractVector{T}) where {T}
    𝕁(assign_q_and_p(v₁, 𝕁.n), assign_q_and_p(v₂, 𝕁.n))
end

(𝕁::PoissonTensor)(qp::QPTOAT) = 𝕁 * qp

# This assigns the right index for the symplectic potential. To be used with `assign_ones_for_poisson_tensor_kernel!`.
function map_index_for_poisson_tensor(i::Int, n::Int)
    if i ≤ n
        return (i, i + n)
    else
        return (i, i - n)
    end
end
