
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
struct PoissonTensor{T, AT} <: AbstractMatrix{T}
    J::AT
    n::Int
end

Base.getindex(𝕁::PoissonTensor, i, j) = getindex(𝕁.J, i, j)

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

Base.:*(::PoissonTensor, v::QPT) = (q = v.p, p = -v.q)
function Base.:*(𝕁::PoissonTensor{T}, v::AbstractArray{T, 3}) where {T}
    _vcat(𝕁(assign_q_and_p(v, 𝕁.n)))
end
function Base.:*(𝕁::PoissonTensor{T}, v::AbstractVector{T}) where {T}
    _vcat(𝕁(assign_q_and_p(v, 𝕁.n)))
end
function Base.:*(𝕁::PoissonTensor{T}, v::AbstractMatrix{T}) where {T}
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
