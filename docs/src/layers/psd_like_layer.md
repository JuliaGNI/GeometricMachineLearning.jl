# PSD-Like Layers and Analytic Layer Inversion

Cotangent-lift [`PSDLayer`](@ref) and [`TransposePSDLayer`](@ref) layers change the dimension of a canonical symplectic vector space using a single block matrix ``\Phi \in \mathrm{St}(n, N)``, while [`inverse_layer`](@ref) constructs the exact analytic inverse (or symplectic transpose) of [`GradientLayerQ`](@ref), [`GradientLayerP`](@ref), [`SymplecticAttentionQ`](@ref), [`SymplecticAttentionP`](@ref), and [`PSDLayer`](@ref) sharing the same parameter `NamedTuple`.

## Library Functions

```@docs
PSDLayer
TransposePSDLayer
```
