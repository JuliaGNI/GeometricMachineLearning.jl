# Linear Symplectic Attention

The attention layer introduced here is an extension of the [Sympnet gradient layer](@ref "SympNet Gradient Layer") to the setting where we deal with time series data. We first have to define a notion of symplecticity for [multi-step methods](@ref "Multi-step methods"). 

This definition is different from [feng1987symplectic, ge1988approximation](@cite), but similar to the definition of volume-preservation in [brantner2024volume](@cite)[^1]. 

[^1]: This definition is also recalled in the section on [volume-preserving attention](@ref "How is structure preserved?").

```@eval
Main.definition(raw"""
A multi-step method ``\varphi\times_T\mathbb{R}^{2n}\to\times_T\mathbb{R}^{2n}`` is called **symplectic** if it preserves the the symplectic product structure, i.e. if ``hat{\varphi}`` is symplectic.""")
```

```@eval
Main.remark(raw"The **symplectic product structure** is the following skew-symmetric non-degenerate bilinear form: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\hat{\mathbb{J}}([z^{(1)}, \ldots, z^{(T)}], [\tilde{z}^{(1)}, \ldots, \tilde{z}^{(T)}]) := \sum_{i=1}^T (z^{(i)})^T\tilde{z}^{(i)}.
" * Main.indentation * raw"```
" * Main.indentation * raw"``\hat{\mathbb{J}}`` is defined through the isomorphism between the product space and the space of big vectors ``\hat{}: \times_\text{($T$ times)}\mathbb{R}^{d}\stackrel{\approx}{\longrightarrow}\mathbb{R}^{dT}``.")
```

In order to construct a symplectic attention mechanism we extend the principle [SympNet gradient layer](@ref "SympNet Gradient Layer"), i.e. we construct scalar functions that only depend on ``[q^{(1)}, \ldots, q^{(T)}]`` or ``[p^{(1)}, \ldots, p^{(T)}]``. The specific choice we make here is the following: 

```math
F(q^{(1)}, q^{(T)}) = \frac{1}{2}\mathrm{Tr}(QAQ^T),
```

where ``Q := [q^{(1)}, \ldots, q^{(T)}]``. We therefore have for the gradient:

```math 
\nabla_Qf = \frac{1}{2}Q(A + A^T) =: Q\bar{A},
```

where ``A\in\mathcal{S}_\mathrm{skew}(T)``. So the map performs:

```math
[q^{(1)}, \ldots, q^{(T)}] \mapsto \left[ \sum_{i=1}^Ta_{1i}q^{(i)}, \ldots, \sum_{i=1}^Ta_{Ti}q^{(i)} \right].
```

Note that there is still a reweighting of the input vectors performed with this linear symplectic attention, like in [standard attention](@ref "Reweighting of the Input Sequence ") and [volume-preserving attention](@ref "Volume-Preserving Attention"), but the crucial difference is that the coefficients ``a`` here are in linear relation to the input vectors, as opposed to the coefficients ``y`` for the [standard and volume-preserving attention layers](@ref "The Attention Layer"), which depend on the input vectors non-linearly. We hence call this attention mechanism *linear symplectic attention* to distinguish it from the standard attention mechanism, which computes reweighting coefficients that depend on the input nonlinearly.

## Library Functions

```@docs; canonical=false
LinearSymplecticAttention
LinearSymplecticAttentionQ
LinearSymplecticAttentionP
```