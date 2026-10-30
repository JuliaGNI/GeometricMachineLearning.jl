# Bilinear and Low-Rank Stiefel Layers

`BilinearStiefelLayer`, `TransposeBilinearStiefelLayer`, `LowRankStiefelLayer`,
`DecodeLowRankStiefelLayer`, and `FastDLRATraceLoss` provide velocity-parity Stiefel
frontends for two-sided phase-space grid fields ``F \in \mathbb{R}^{N_x \times N_v}`` and
dynamical low-rank approximation (DLRA) states ``(X, S, V)``.

Their ``(q, p)`` is canonical only for the structure ``\mathbb{J}`` that they declare on the
parity coordinates. A frontend that is linear in ``f`` is not a Poisson map for the Vlasov
bracket: with ``L^2``-orthonormal even and odd bases, the bracket linearised at ``f_0`` gives a
singular ``\{Q, P\}``, not the identity.

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
