
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
function SymplecticPotential(backend, n2::Int, T::DataType=Float64)
    @assert iseven(n2)
    n = n2÷2
    J = KernelAbstractions.zeros(backend, T, 2*n, 2*n)
    assign_ones_for_symplectic_potential! = assign_ones_for_symplectic_potential_kernel!(backend)
    assign_ones_for_symplectic_potential!(J, n, ndrange=n)
    J
end

SymplecticPotential(n::Int, T::DataType=Float64) = SymplecticPotential(CPU(), n, T)
SymplecticPotential(bakend, T::DataType, n::Int) = SymplecticPotential(backend, n, T)

SymplecticPotential(T::DataType, n::Int) = SymplecticPotential(n, T)

@kernel function assign_ones_for_symplectic_potential_kernel!(J::AbstractMatrix{T}, n::Int) where T
    i = @index(Global)
    J[map_index_for_symplectic_potential(i, n)...] = i ≤ n ? one(T) : -one(T)
end

"""
This assigns the right index for the symplectic potential. To be used with `assign_ones_for_symplectic_potential_kernel!`.
"""
function map_index_for_symplectic_potential(i::Int, n::Int)
    if i ≤ n
        return (i, i+n)
    else
        return (i, i-n)
    end
end