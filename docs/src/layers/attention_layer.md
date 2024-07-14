# The Attention Layer

The *attention* mechanism was originally developed for image and natural language processing (NLP) tasks. It is motivated by the need to handle time series data in an efficient way[^1]. Its essential idea is to compute correlations between vectors in input sequences. So given two sequences 

```math
(z_q^{(1)}, z_q^{(2)}, \ldots, z_q^{(T)}) \text{ and } (z_p^{(1)}, z_p^{(2)}, \ldots, z_p^{(T)}),
```
an attention mechanism computes pair-wise correlations between all combinations of two input vectors from these sequences. In [bahdanau2014neural](@cite) "additive" attention is used to compute such correlations: 

[^1]: *Recurrent neural networks* have the same motivation. 

```math
(z_q, z_k) \mapsto v^T\sigma(Wz_q + Uz_k), 
```

where ``z_q, z_k \in \mathbb{R}^d`` are elements of the input sequences. The learnable parameters are ``W, U \in \mathbb{R}^{n\times{}d}`` and ``v \in \mathbb{R}^n``.

However *multiplicative attention* (see e.g. [vaswani2017attention](@cite)) is more straightforward to interpret and cheaper to handle computationally: 

```math
(z_q, z_k) \mapsto z_q^TWz_k,
```

where ``W \in \mathbb{R}^{d\times{}d}`` is a learnable weight matrix with respect to which correlations are computed as scalar products. Regardless of the type of attention used, they all try to compute correlations among input sequences on whose basis further computation is performed. Given two input sequences ``Z_q = (z_q^{(1)}, \ldots, z_q^{(T)})`` and ``Z_k = (z_k^{(1)}, \ldots, z_k^{(T)})``, we can arrange the various correlations into a *correlation matrix* ``C\in\mathbb{R}^{T\times{}T}`` with entries ``[C]_{ij} = \mathtt{attention}(z_q^{(i)}, z_k^{(j)})``. In the case of multiplicative attention this matrix is just ``C = Z^TWZ``.

## Reweighting of the input sequence 

In `GeometricMachineLearning` we always compute *self-attention*, meaning that the two input sequences ``Z_q`` and ``Z_k`` are the same, i.e. ``Z = Z_q = Z_k``.[^2]

[^2]: [Multihead attention](multihead_attention_layer.md) also falls into this category. Here the input ``Z`` is multiplied from the left with several *projection matrices* ``P^Q_i`` and ``P^K_i``, where ``i`` indicates the *head*. For each head we then compute a correlation matrix ``(P^Q_i Z)^T(P^K Z)``. 

This is then used to reweight the columns in the input sequence ``Z``. For this we first apply a nonlinearity ``\sigma`` onto ``C`` and then multiply ``\sigma(C)`` onto ``Z`` from the right, i.e. the output of the attention layer is ``Z\sigma(C)``. So we perform the following mappings:

```math
Z \xrightarrow{\mathrm{correlations}} C(Z) =: C \xrightarrow{\sigma} \sigma(C) \xrightarrow{\text{right multiplication}} Z \sigma(C).
```


After the right multiplication the outputs is of the following form: 

```math 
    [\sum_{i=1}^Tp^{(1)}_iz^{(i)}, \ldots, \sum_{i=1}^Tp^{(T)}_iz^{(i)}],
```
for ``p^{(i)} = [\sigma(C)]_{\bullet{}i}``. What is *learned* during training are ``T`` different linear combinations of the input vectors, where the coefficients ``p^{(i)}_j`` in these linear combinations depend on the input ``Z`` nonlinearly. 

## Volume-Preserving Attention

The [`VolumePreservingAttention`](@ref) layer (and the activation function ``\sigma`` defined for it) in `GeometricMachineLearning` was specifically designed to apply it to data coming from physical systems that can be described through a divergence-free or a symplectic vector field. 
Traditionally the nonlinearity in the attention mechanism is a softmax[^3] [vaswani2017attention](@cite) and the self-attention layer performs the following mapping: 

[^3]: The softmax acts on the matrix ``C`` in a vector-wise manner, i.e. it operates on each column of the input matrix ``C = [c^{(1)}, \ldots, c^{(T)}]``. The result is a sequence of probability vectors ``[p^{(1)}, \ldots, p^{(T)}]`` for which ``\sum_{i=1}^Tp^{(j)}_i=1\quad\forall{}j\in\{1,\dots,T\}.``

```math
Z := [z^{(1)}, \ldots, z^{(T)}] \mapsto Z\mathrm{softmax}(Z^TWZ).
```

The softmax activation acts vector-wise, i.e. if we supply it with a matrix ``C`` as input it returns: 

```math 
\mathrm{softmax}(C) = [\mathrm{softmax}(c_{\bullet{}1}), \ldots, \mathrm{softmax}(c_{\bullet{}T})].
```

The output of a softmax is a *probability vector* (also called *stochastic vector*) and the matrix ``P = [p^{(1)}, \ldots, p^{(T)}]``, where each column is a probability vector, is sometimes referred to as a *stochastic matrix* (see [jacobs1992discrete](@cite)). This attention mechanism finds application in *transformer neural networks* [vaswani2017attention](@cite). The problem with this matrix from a geometric point of view is that all the columns are independent of each other and the nonlinear transformation could in theory produce a stochastic matrix for which all columns are identical and thus lead to a loss of information. So the softmax activation function is inherently non-geometric. We visualize this with the figure below: 

![](../tikz/convex_recombination.png)

So the ``y`` coefficients responsible for producing the first output vector are independent from those producing the second output vector etc., they have the condition ``\sum_{i=1}^Ty^{(j)}_iz_\mu^{(i)}`` for each column ``j`` imposed on them, but the coefficients for two different columns are independent of each other.

Besides the traditional attention mechanism `GeometricMachineLearning` therefore also has a volume-preserving transformation that fulfills a similar role. There are two approaches implemented to realize similar transformations. Both of them however utilize the *Cayley transform* to produce orthogonal matrices ``\sigma(C)`` instead of stochastic matrices. For an orthogonal matrix ``\Sigma`` we have ``\Sigma^T\Sigma = \mathbb{I}``, so all the columns are linearly independent which is not necessarily true for a stochastic matrix ``P``. In the following we explain how this new activation function is implemented. First we need to briefly discuss the *Cayley transform*. 

### The Cayley transform 

The Cayley transform maps from skew-symmetric matrices to orthonormal matrices[^4]. It takes the form:

[^4]: A matrix ``A`` is skew-symmetric if ``A = -A^T`` and a matrix ``B`` is orthonormal if ``B^TB = \mathbb{I}``. The orthonormal matrices form a Lie group, i.e. the set of orthonormal matrices can be endowed with the structure of a differential manifold and this set also satisfies the group axioms. The corresponding Lie algebra are the skew-symmetric matrices and the Cayley transform is a so-called retraction in this case. For more details consult e.g. [hairer2006geometric](@cite) and [absil2008optimization](@cite).

```math 
\mathrm{Cayley}: A \mapsto (\mathbb{I} - A)(\mathbb{I} + A)^{-1}.
```

We can easily check that ``\mathrm{Cayley}(A)`` is orthogonal if ``A`` is skew-symmetric. For this consider ``\varepsilon \mapsto A(\varepsilon)\in\mathcal{S}_\mathrm{skew}`` with ``A(0) = \mathbb{I}`` and ``A'(0) = B``. Then we have: 

```math
\frac{\delta(\mathrm{Cayley}(A)^T\mathrm{Cayley}(A))}{\delta{}A} = \frac{d}{d\varepsilon}|_{\varepsilon=0} \mathrm{Cayley}(A(\varepsilon))^T \mathrm{Cayley}(A(\varepsilon)) = A'(0)^T + A'(0) = \mathbb{O},
```

So ``\mathrm{Cayley}(A)^T\mathrm{Cayley}(A)`` remains unchanged among ``\varepsilon``. In order to use the Cayley transform as an activation function we further need a mapping from the input ``Z`` to a skew-symmetric matrix. This is realized in two ways in `GeometricMachineLearning`: via a scalar-product with a skew-symmetric weighting and via a scalar-product with an arbitrary weighting.

### First approach: scalar products with a skew-symmetric weighting

For this the attention layer is modified in the following way: 

```math 
Z := [z^{(1)}, \ldots, z^{(T)}] \mapsto Z\sigma(Z^TAZ),
```
where ``\sigma(C)=\mathrm{Cayley}(C)`` and ``A`` is a matrix of type [`SkewSymMatrix`](@ref) that is learnable, i.e. the parameters of the attention layer are stored in ``A``.

### Second approach: scalar products with an arbitrary weighting

For this approach we compute correlations between the input vectors based on scalar product with an arbitrary weighting. This arbitrary ``T\times{}T`` matrix ``A`` constitutes the learnable parameters of the attention layer. The correlations we consider here are based on: 

```math
(z^{(2)})^TAz^{(1)}, (z^{(3)})^TAz^{(1)}, \ldots, (z^{(d)})^TAz^{(1)}, (z^{(3)})^TAz^{(2)}, \ldots, (z^{(d)})^TAz^{(2)}, \ldots, (z^{(d)})^TAz^{(d-1)}.
```

So we consider correlations ``(z^{(i)})^Tz^{(j)}`` for which ``i > j``. We now arrange these correlations into a skew-symmetric matrix: 

```math
C = \begin{bmatrix}
        0               & -(z^{(2)})^TAz^{(1)} & -(z^{(3)})^TAz^{(1)} &     \ldots & -(z^{(d)})^TAz^{(1)} \\
    (z^{(2)})^TAz^{(1)} &       0              & -(z^{(3)})^TAz^{(2)} &     \ldots & -(z^{(d)})^TAz^{(2)} \\
    \ldots              &       \ldots         &        \ldots        &     \ldots &    \ldots             \\
    (z^{(d)})^TAz^{(1)} & (z^{(d)})^TAz^{(2)}  & (z^{(d)})^TAz^{(3)}  &     \ldots &        0               
\end{bmatrix}.
```

This correlation matrix can now again be used as an input for the Cayley transform to produce an orthogonal matrix. Mathematically this is also equivalent to first computing all correlations ``Z^TAZ`` and then mapping the lower triangular to the upper triangular and negating these elements. This is visualized below: 

```@example
Main.include_graphics("../tikz/skew_sym_mapping")
```

Internally `GeometricMachineLearning` computes this more efficiently with the function [`GeometricMachineLearning.tensor_mat_skew_sym_assign`](@ref).

## How is structure preserved? 

In order to discuss *how structure is preserved* we first have to define what *structure* we mean precisely. This structure is strongly inspired by traditional *multi-step methods* (see [feng1998step](@cite)). We now define what volume preservation means for the product space ``\mathbb{R}^{d}\times\cdots\times\mathbb{R}^{d}\equiv\times_\text{$T$ times}\mathbb{R}^{d}``.

Consider an isomorphism ``\hat{}: \times_\text{($T$ times)}\mathbb{R}^{d}\stackrel{\approx}{\longrightarrow}\mathbb{R}^{dT}``. Specifically, this isomorphism takes the form:
```math
Z =  \left[\begin{array}{cccc}
            z_1^{(1)} &  z_1^{(2)} & \quad\cdots\quad & z_1^{(T)} \\
            z_2^{(1)} &  z_2^{(2)} & \cdots & z_2^{(T)} \\
            \cdots &  \cdots & \cdots & \cdots \\
            z_d^{(1)} & z_d^{(2)} & \cdots & z_d^{(T)}
            \end{array}\right] \mapsto 
            \left[\begin{array}{c}  z_1^{(1)} \\ z_1^{(2)} \\ \cdots \\ z_1^{(T)} \\ z_2^{(1)} \\ \cdots \\ z_d^{(T)} \end{array}\right] =: Z_\mathrm{vec},
```

so we arrange the rows consecutively into a vector. The inverse of ``Z \mapsto \hat{Z} `` we refer to as ``Y \mapsto \tilde{Y}``. In the following we also write ``\hat{\varphi}`` for the mapping ``\,\hat{}\circ\varphi\circ\tilde{}\,``.

```@eval
Main.definition(raw"We say that a mapping ``\varphi: \times_\text{$T$ times}\mathbb{R}^{d} \to \times_\text{$T$ times}\mathbb{R}^{d}`` is **volume-preserving** if the associated ``\hat{\varphi}`` is volume-preserving.")
```

In the transformed coordinate system (in terms of the vector ``Z_\mathrm{vec}`` defined above) this is equivalent to multiplication by a sparse matrix ``\tilde\Lambda(Z)`` from the left:

```math
    \tilde{\Lambda}(Z) Z_\mathrm{vec} :=
    \begin{pmatrix}
    \Lambda(Z) & \mathbb{O} & \cdots  & \mathbb{O} \\
    \mathbb{O} & \Lambda(Z) & \cdots & \mathbb{O} \\
    \cdots & \cdots & \ddots & \cdots \\ 
    \mathbb{O} & \mathbb{O} & \cdots & \Lambda(Z) \\
    \end{pmatrix}
    \left[\begin{array}{c}  z_1^{(1)} \\ z_1^{(2)} \\ \ldots \\ z_1^{(T)} \\ z_2^{(1)} \\ \ldots \\ z_d^{(T)} \end{array}\right] .
```

``\tilde{\Lambda}(Z)`` is easily shown to be an orthogonal matrix and a symplectic matrix, i.e. it satisfies

```math
\tilde{\Lambda}(Z)^T\tilde{\Lambda}(Z) = \mathbb{I}
```

and

```math
\tilde{\Lambda}(Z)^T\mathbb{J}\tilde{\Lambda}(Z) = \mathbb{J}.
```


## Historical Note 

Attention was used before the transformer was introduced, but mostly in connection with *recurrent neural networks* (see [luong2015effective](@cite) and [bahdanau2014neural](@cite)). 

## Library Functions

```@docs; canonical = false
tensor_mat_skew_sym_assign
MultiHeadAttention
VolumePreservingAttention
```

## References 

```@bibliography
Pages = []
Canonical = false 

bahdanau2014neural
luong2015effective
```