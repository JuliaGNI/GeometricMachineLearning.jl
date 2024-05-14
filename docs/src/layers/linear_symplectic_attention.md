# Linear Symplectic Attention

The attention layer introduced here is an extension of the [Sympnet gradient layer](@ref "SympNet Gradient Layer") to the setting where we deal with time series data. We first have to define a notion of symplecticity for [multi-step methods](@ref "Multi-step methods"). 

This definition is essentially taken from [feng1987symplectic, ge1988approximation](@cite) and similar to the definition of volume-preservation in [brantner2024volume](@cite). 

```@eval
Main.definition(raw"""
A multi-step method ``\times_T\mathbb{R}^{2n}\to\times_T\mathbb{R}^{2n}`` is called **symplectic** if it preserves the the symplectic product structure.
""")
```

The *symplectic product structure* is the following skew-symmetric non-degenerate bilinear form: 

```math
\mathbb{J}([z^{(1)}, \ldots, z^{(T)}], [\tilde{z}^{(1)}, \ldots, \tilde{z}^{(T)}]) := \sum_{i=1}^T (z^{(i)})^T\tilde{z}^{(i)}.
```

In order to construct a symplectic attention mechanism we extend the principle [SympNet gradient layer](@ref "SympNet Gradient Layer"), i.e. we construct scalar functions that only depend on ``[q^{(1)}, \ldots, q^{(T)}]`` or ``[p^{(1)}, \ldots, p^{(T)}]``. The specific choice we make here is the following: 

```math
F(q^{(1)}, q^{(T)}) = \frac{1}{2}\mathrm{Tr}(QAQ^T),
```

where ``Q := [q^{(1)}, \ldots, q^{(T)}]``. We therefore have for the gradient:

```math 
\nabla_Qf = \frac{1}{2}Q(A + A^T) =: Q\bar{A},
```

where ``A\in\mathcal{S}_\mathrm{skew}(T). So the map performs:

```math
[q^{(1)}, \ldots, q^{(T)}] \mapsto \left[ \sum_{i=1}^Ta_{1i}q^{(i)}, \ldots, \sum_{i=1}^Ta_{Ti}q^{(i)} \right].
```

## Library Functions

```@docs; canonical=false
LinearSymplecticAttention
LinearSymplecticAttentionQ
LinearSymplecticAttentionP
```