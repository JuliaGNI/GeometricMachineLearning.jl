
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
"""
function SymplecticMatrix(n::Int, T::DataType=Float64)
    BandedMatrix((n => ones(T,n), -n => -ones(T,n)), (2n,2n))
end

SymplecticMatrix(T::DataType, n::Int) = SymplecticMatrix(n, T)
