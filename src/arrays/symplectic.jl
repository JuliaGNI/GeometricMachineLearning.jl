
@doc raw"""

`SymplecticPotential(n)`

Returns a symplectic matrix of size 2n x 2n

```math
\begin{pmatrix}
\mathbb{O} & \mathbb{I} \\
\mathbb{O} & -\mathbb{I} \\
\end{pmatrix}
```
"""
struct SymplecticPotential{T, AT} <: AbstractMatrix{T}
    J::AT
    n::Int
end

Base.getindex(𝕁::SymplecticPotential, i, j) = getindex(𝕁.J, i, j)

Base.size(𝕁::SymplecticPotential) = size(𝕁.J)

function SymplecticPotential(backend::Backend, n2::Int, T::DataType)
    @assert iseven(n2)
    n = n2÷2
    J = KernelAbstractions.zeros(backend, T, 2*n, 2*n)
    assign_ones_for_symplectic_potential! = assign_ones_for_symplectic_potential_kernel!(backend)
    assign_ones_for_symplectic_potential!(J, n, ndrange=n2)
    
    SymplecticPotential{T, typeof(J)}(J, n)
end

SymplecticPotential(n2::Int, T::DataType) = SymplecticPotential(CPU(), n2, T)

SymplecticPotential(n2::Int) = SymplecticPotential(n2, Float64)

SymplecticPotential(backend::Backend, n2::Int) = SymplecticPotential(backend, n2, Float32)

SymplecticPotential(backend::CPU, n2::Int) = SymplecticPotential(backend, n2, Float64)

@kernel function assign_ones_for_symplectic_potential_kernel!(J::AbstractMatrix{T}, n::Int) where T
    i = @index(Global)
    J[map_index_for_symplectic_potential(i, n)...] = i ≤ n ? one(T) : -one(T)
end

Base.:*(::SymplecticPotential{T}, v::NamedTuple{(:q, :p), Tuple{AT, AT}}) where {T, AT <: AbstractVecOrMat{T}} = (q = v.p, p = -v.q)

function _vcat(v::NamedTuple{(:q, :p), Tuple{AT, AT}}) where {AT <: AbstractArray}
    vcat(v.q, v.p)
end

Base.:*(𝕁::SymplecticPotential{T}, v::AbstractVector{T}) where T = _vcat(𝕁 * assign_q_and_p(v, 𝕁.n))
Base.:*(𝕁::SymplecticPotential{T}, v::AbstractMatrix{T}) where T = _vcat(𝕁 * assign_q_and_p(v, 𝕁.n))


function (𝕁::SymplecticPotential{T})(v₁::NT, v₂::NT) where {T, AT <: AbstractVector{T}, NT <: NamedTuple{(:q, :p), Tuple{AT, AT}}}
    v₁.q' * v₂.p - v₁.p' * v₂.q
end

function (𝕁::SymplecticPotential{T})(v₁::AbstractVector{T}, v₂::AbstractVector{T}) where T 
    𝕁(assign_q_and_p(v₁, 𝕁.n), assign_q_and_p(v₂, 𝕁.n))
end

"""
This assigns the right index for the symplectic potential. To be used with `assign_ones_for_symplectic_potential_kernel!`.
"""
function map_index_for_symplectic_potential(i::Int, n::Int)
    if i ≤ n
        return (i, i + n)
    else
        return (i, i - n)
    end
end