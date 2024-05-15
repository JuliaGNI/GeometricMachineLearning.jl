# Linear Symplectic Transformer

The linear symplectic transformer consists of a combination of [linear symplectic attention](@ref "Linear Symplectic Attention") and [gradient](@ref "SympNet Gradient Layer") layers and is visualized below: 

```@example
Main.include_graphics("../tikz/linear_symplectic_transformer"; caption=raw"Visualization of the linear symplectic transformer architecutre. `n_sympnet` refers to the number of SympNet layers (`n_sympnet=2` in this figure) and $L$ refers to the number of transformer blocks (`L=1` in this figure).") # hide
```


```@docs; canonical=false
LinearSymplecticTransformer
```