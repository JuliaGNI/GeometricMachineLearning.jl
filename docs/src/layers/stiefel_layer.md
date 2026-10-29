# Bilinear and Low-Rank Stiefel Layers

`BilinearStiefelLayer`, `TransposeBilinearStiefelLayer`, `LowRankStiefelLayer`,
`DecodeLowRankStiefelLayer`, and `FastDLRATraceLoss` provide canonical velocity-parity
Stiefel frontends for two-sided phase-space grid fields ``F \in \mathbb{R}^{N_x \times N_v}``
and dynamical low-rank approximation (DLRA) states ``(X, S, V)``.

## Bilinear Grid and Spline Stiefel Layers

```@docs
BilinearStiefelLayer
TransposeBilinearStiefelLayer
solve!(::BilinearStiefelLayer, ::NamedTuple, ::AbstractArray{<:Any, 3})
```

## Dynamical Low-Rank Stiefel Layers and Trace Loss

```@docs
LowRankStiefelLayer
DecodeLowRankStiefelLayer
FastDLRATraceLoss
```
