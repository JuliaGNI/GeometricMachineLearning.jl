# Linear Symplectic Attention

The attention layer introduced here can be seen as an extension of the [SympNet gradient layer](@ref "SympNet Gradient Layer") to the setting where we deal with time series data. Before we introduce the [`LinearSymplecticAttention`](@ref) layer we first define a notion of symplecticity for [multi-step methods](@ref "Multi-step methods"). 

This definition is different from [feng1987symplectic, ge1988approximation](@cite), but similar to the [definition of volume-preservation for product spaces](@ref "How is Structure Preserved?") in [brantner2024volume](@cite).

```@eval
Main.definition(raw"""
A multi-step method ``\varphi\times_T\mathbb{R}^{2n}\to\times_T\mathbb{R}^{2n}`` is called **symplectic** if it preserves the the symplectic product structure, i.e. if ``\hat{\varphi}`` is symplectic.""")
```

```@eval
Main.remark(raw"The **symplectic product structure** is the following skew-symmetric non-degenerate bilinear form: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\hat{\mathbb{J}}([z^{(1)}, \ldots, z^{(T)}], [\tilde{z}^{(1)}, \ldots, \tilde{z}^{(T)}]) := \sum_{i=1}^T (z^{(i)})^T\mathbb{J}_{2n}\tilde{z}^{(i)}.
" * Main.indentation * raw"```
" * Main.indentation * raw"``\hat{\mathbb{J}}`` is defined through the isomorphism between the product space and the space of big vectors 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\hat{}: \times_\text{($T$ times)}\mathbb{R}^{d}\stackrel{\approx}{\longrightarrow}\mathbb{R}^{dT},
" * Main.indentation * raw"```
" * Main.indentation * raw"so we induce the symplectic structure on the product space through the pullback of this isomorphism.")
```

In order to construct a symplectic attention mechanism we extend the principle behind the [SympNet gradient layer](@ref "SympNet Gradient Layer"), i.e. we construct scalar functions that only depend on ``[q^{(1)}, \ldots, q^{(T)}]`` or ``[p^{(1)}, \ldots, p^{(T)}]``. The specific choice we make here is the following: 

```math
F(q^{(1)}, \ldots, q^{(T)}) = \frac{1}{2}\mathrm{Tr}(QAQ^T),
```

where ``Q := [q^{(1)}, \ldots, q^{(T)}]`` is the concatenation of the vectors into a matrix. We therefore have for the gradient:

```math 
\nabla_Qf = \frac{1}{2}Q(A + A^T) =: Q\bar{A},
```

where ``\bar{A}\in\mathcal{S}_\mathrm{sym}(T)`` is a symmetric matrix. So the map performs:

```math
[q^{(1)}, \ldots, q^{(T)}] \mapsto \left[ \sum_{i=1}^Ta_{1i}q^{(i)}, \ldots, \sum_{i=1}^Ta_{Ti}q^{(i)} \right] \text{ for } a_{ji} = [\bar{A}]_{ji}.
```

Note that there is still a reweighting of the input vectors performed with this linear symplectic attention, like in [standard attention](@ref "Reweighting of the Input Sequence ") and [volume-preserving attention](@ref "Volume-Preserving Attention"), but the crucial difference is that the coefficients ``a_{ji}`` here are fixed and not computed as the result of a softmax or a [Cayley transform](@ref "The Cayley Transform"). We hence call this attention mechanism *linear symplectic attention* as it performs a linear reweighting of the input vectors. We distinguish it from the [standard attention mechanism](@ref "The Attention Layer"), which computes coefficients that depend on the input nonlinearly.

## Library Functions

```@docs
LinearSymplecticAttention
LinearSymplecticAttentionQ
LinearSymplecticAttentionP
```

```@raw latex
\section*{Chapter Summary}

In this chapter we discussed various neural network layers and the corresponding application interface in \texttt{GeometricMachineLearning}. Some of these layers constitute novel work (like the volume-preserving attention layer and the linear symplectic layer) and others were established before (such as SympNet layers and multihead attention). Volume-preserving attention and linear symplectic attention were designed as a modification of standard attention in order to imbue the corresponding neural network with structure (volume preservation and symplecticity respectively). Volume-preserving attention was achieved by exchanging the softmax activation function with a \textit{Cayley activation function}, but otherwise keeping the operations the same, i.e. we still perform \textit{multiplicative attention}. In order to make the attention layer symplectic we had to impose more severe restrictions on it: the attention mechanism does not compute a scalar product in this case. We therefore call it \textit{linear}.

In the next chapter we use these neural network layers to build \textit{neural network architectures}.
```

```@raw html
<!--
```

```@bibliography
Pages = []
Canonical = Pages

jin2020sympnets
bahdanau2014neural
luong205effective
vaswani2017attention
```

```@raw html
-->
```