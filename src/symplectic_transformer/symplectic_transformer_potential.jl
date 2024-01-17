@doc raw"""
The potential used for implementing the symplectic transformer. Its precise form is: 

```math 
F(Z) = \frac{1}{2}\mathrm{Tr}(Z^T\mathrm{softmax}(Z^TAZ)Z).
```

The idea behind it is that its derivative looks similar to a single-head attention layer and we can then construct a symplectic transformer in a similar way to how it is done with SympNets.
"""
symplectic_transformer_potential(Z::AbstractMatrix{T}, A::AbstractMatrix{T}) where T = T(.5) * tr(Z * softmax(Z' * A * Z) * Z')