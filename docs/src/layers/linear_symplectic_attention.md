# Linear Symplectic Attention 

The attention layer introduced here is an extension of the [Sympnet gradient layer](@ref symp_grad_layer) to the setting where we deal with time series data. We first have to define a notion of symplecticity for multi-step methods. 

This definition is essentially taken from [feng1987symplectic, ge1988approximation](@cite) and similar to the definition of volume-preservation in [brantner2024volume](@cite). 

```@eval
Main.definition(raw"""
A multi-step method is called **symplectic** if it preserves 
""")
```