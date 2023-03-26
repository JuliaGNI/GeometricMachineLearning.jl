
@doc raw"""

    `SymplecticMatrix(n)`

Returns a symplectic matrix of size 2n x 2n

```math
\begin{pmatrix}
0 & & & 1 & & & \\
& \ddots & & & \ddots & & \\
& & 0 & & & 1 \\
-1 & & & 0 & & & \\
& \ddots & & & \ddots & & \\
& & -1 & & 0 & \\
\end{pmatrix}
```

    `SymplecticProjection(N,n)`
Returns the symplectic projection matrix E of the Stiefel manifold, i.e. π: Sp(2N) → Sp(2n,2N), A ↦ AE

"""
function SymplecticMatrix(n::Int, T::DataType=Float64)
    BandedMatrix((n => ones(T,n), -n => -ones(T,n)), (2n,2n))
end

SymplecticMatrix(T::DataType, n::Int) = SymplecticMatrix(n, T)

@doc raw"""
```math
\begin{pmatrix}
I & 0 \\
0 & 0 \\
0 & I \\
0 & 0 \\
\end{pmatrix}
```
"""

struct SymplecticProjection{T} <: AbstractMatrix{T}
    N::Int
    n::Int
    SymplecticProjection(N, n, T = Float64) = new{T}(N,n)
end

function Base.getindex(E::SymplecticProjection,i,j)
    if i ≤ E.n
        if j == i 
            return 1.
        end
        return 0.
    end
    if j > E.n
        if (j-E.n) == (i-E.N)
            return 1.
        end
        return 0.
    end
    return 0.
end


Base.parent(E::SymplecticProjection) = (E.N,E.n)
Base.size(E::SymplecticProjection) = (2*E.N,2*E.n)
