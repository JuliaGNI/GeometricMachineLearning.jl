# Linear Symplectic Transformer

The linear symplectic transformer consists of a combination of [linear symplectic attention](@ref "Linear Symplectic Attention") and [gradient](@ref "SympNet Gradient Layer") layers and is visualized below: 

```@example
Main.include_graphics("../tikz/linear_symplectic_transformer"; caption = raw"Visualization of the linear symplectic transformer architecutre. \texttt{n\_sympnet} refers to the number of SympNet layers (\texttt{n\_sympnet=2} in this figure) and \texttt{L} refers to the number of transformer blocks (\texttt{L=1} in this figure).", width = .3) # hide
```

## Library Functions

```@docs; canonical=false
LinearSymplecticTransformer
```