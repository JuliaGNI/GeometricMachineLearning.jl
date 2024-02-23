# The Attention Layer

The *attention* mechanism was originally applied for image and natural language processing (NLP) tasks. It is motivated by the need to handle time series data in an efficient way[^1]. In [bahdanau2014neural](@cite) ``additive'' attention is used to compute the correlations between two vectors in a time series: 

[^1]: *Recurrent neural networks* have the same motivation. 

```math
(z_q, z_k) \mapsto v^T\sigma(Wz_q + Uz_k), 
```

where ``z_q, z_k \in \mathbb{R}^d``, ``W, U \in \mathbb{R}^{n\times{}d}`` and ``v \in \mathbb{R}^n``.

However *multiplicative attention* is more straightforward to interpret and cheaper to handle computationally: 

```math
(z_q, z_k) \mapsto z_q^TWz_k.
```

Regardless of the type of attention used, they all try to compute correlations among input sequences on whose basis further computation is performed. Given two input sequences ``Z_q = (z_q^{(1)}, \ldots, z_q^{(T)})`` and ``Z_k = (z_k^{(1)}, \ldots, z_k^{(T)})``, we can arrange the results of computing the various correlations into a *correlation matrix* ``C\in\mathbb{R}^{T\times{}T}`` with entries ``[C]_{ij} = \mathtt{attention}(z_q^{(i)}, z_k^{(j)}``.

## Reweighting of the input sequence 

In `GeometricMachineLearning` we always compute *self-attention*, meaning that the two input sequences ``Z_q`` and ``Z_k`` are the same, i.e. ``Z = Z_q = Z_k``[^2].

[^2]: [Multihead attention](multihead_attention.md) also falls into this category. Here the input ``Z`` is multiplied from the left with several *projection matrices* ``P^Q_i`` and ``P^K_i``, where ``i`` indicates the *head*. For each head we then compute a correlation matrix ``(P^Q_i Z)^T(P^K Z)``. 

This is then used to reweight the columns in the input sequence ``Z`` by multiplying ``C`` onto it from the right after we applied a nonlinearity ``\sigma``, i.e. the output of the attention layer is ``Z\sigma(C)``. This means that the final output of the attention layer is: 

```math 
    [\sum_{i=1}^Tp^{(1)}_iz^{(i)}, \ldots, \sum_{i=1}^Tp^{(T)}_iz^{(i)}],
```
for ``p^{(i)} = [\sigma(C)]_{\bullet{}i}.

## `VolumePreservingAttention` in `GeometricMachineLearning`

The attention layer (and the activation function ``sigma`` defined for it) in `GeometricMachineLearning` was specifically designed to apply transformers to data coming from specific physical systems, i.e. systems that can be described through a divergence-free or a symplectic vector field. 
Traditionally the nonlinearity in the attention mechanism is a softmax (see [vaswani2017attention](@cite)) and the entire self-attention layer performs the following mapping: 

```math
Z := [z^{(1)}, \ldots, z^{(T)}] \mapsto Z\mathrm{softmax}(Z^TWZ).
```

The idea behind this is that we can perform a nonlinear reweighting of the columns of ``Z`` by multiplying with a ``Z``-dependent matrix from the right and therefore take the sequential nature of the data into account (which is not possible with standard feedforward neural networks). This attention mechanism finds application in *transformer neural networks*.

The softmax acts on the matrix ``C`` in a vector-wise manner, i.e. it operates on each column of an input matrix $A = [a_1, \ldots, a_T]$. The result is a sequence of probability vectors $[p^{(1)}, \ldots, p^{(T)}]$ for which 

```math
\sum_{i=1}^Tp^{(j)}_i=1\quad\forall{}j\in\{1,\dots,T\}.
```

What we want to construct is a symplectic transformation that is *transformer-like*. For this we modify the attention layer the following way: 

```math 
Z := [z^{(1)}, \ldots, z^{(T)}] \mapsto Z\sigma((P^QZ)^T(P^KZ)),
```
where $\sigma(A)=\exp(\mathtt{upper\_triangular{\_asymmetrize}}(A))$ and 

```math
[\mathtt{upper\_triangular\_asymmetrize}(A)]_{ij} = \begin{cases} a_{ij} & \text{if $i<j$}  \\ -a_{ji} & \text{if $i>j$} \\ 0 & \text{else.}\end{cases}
```

This has as a consequence that the matrix $\Lambda(Z) := \sigma((P^QZ)^T(P^KZ))$ is orthonormal and hence preserves an *extended symplectic structure*. To make this more clear, consider that the transformer maps sequences of vectors to sequences of vectors, i.e. $V\times\cdots\times{}V \ni [z^1, \ldots, z^T] \mapsto [\hat{z}^1, \ldots, \hat{z}^T]$. We can define a symplectic structure on $V\times\cdots\times{}V$ by rearranging $[z^1, \ldots, z^T]$ into a vector. We do this in the following way: 

```math
\tilde{Z} = \begin{pmatrix} q^{(1)}_1 \\ q^{(2)}_1 \\ \cdots \\ q^{(T)}_1 \\ q^{(1)}_2 \\ \cdots \\ q^{(T)}_d \\ p^{(1)}_1 \\ p^{(2)}_1 \\ \cdots \\ p^{(T)}_1 \\ p^{(1)}_2 \\ \cdots \\ p^{(T)}_d \end{pmatrix}.
```

The symplectic structure on this big space is then: 

```math
\mathbb{J}=\begin{pmatrix}
    \mathbb{O}_{dT} & \mathbb{I}_{dT} \\
    -\mathbb{I}_{dT} & \mathbb{O}_{dT}
\end{pmatrix}.
```

Multiplying with the matrix $\Lambda(Z)$ from the right onto $[z^1, \ldots, z^T]$ corresponds to applying the sparse matrix 

```math
\tilde{\Lambda}(Z)=\left[
\begin{array}{ccc}
   \Lambda(Z) & \cdots & \mathbb{O}_T \\
   \vdots & \ddots & \vdots \\
   \mathbb{O}_T & \cdots & \Lambda(Z) 
   \end{array}
\right]
```

from the left onto the big vector. 


## Historical Note 

Attention was used before, but always in connection with **recurrent neural networks** (see (Luong et al, 2015) and (Bahdanau et al, 2014)). 


## References 

```@bibliography
Pages = []
Canonical = false 

bahdanau2014neural
luong2015effective
```