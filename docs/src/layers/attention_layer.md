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

Regardless of the type of attention used, they all try to compute correlations among input sequences on whose basis further computation is performed. Given two input sequences ``Z_q = (z_q^{(1)}, \ldots, z_q^{(T)})`` and ``Z_k = (z_k^{(1)}, \ldots, z_k^{(T)})``, we can arrange the various correlations into a *correlation matrix* ``C\in\mathbb{R}^{T\times{}T}`` with entries ``[C]_{ij} = \mathtt{attention}(z_q^{(i)}, z_k^{(j)})``. In the case of multiplicative attention this matrix is just ``C = Z^TWZ``.

## Reweighting of the input sequence 

In `GeometricMachineLearning` we always compute *self-attention*, meaning that the two input sequences ``Z_q`` and ``Z_k`` are the same, i.e. ``Z = Z_q = Z_k``.[^2]

[^2]: [Multihead attention](multihead_attention_layer.md) also falls into this category. Here the input ``Z`` is multiplied from the left with several *projection matrices* ``P^Q_i`` and ``P^K_i``, where ``i`` indicates the *head*. For each head we then compute a correlation matrix ``(P^Q_i Z)^T(P^K Z)``. 

This is then used to reweight the columns in the input sequence ``Z``. For this we first apply a nonlinearity ``\sigma`` onto ``C`` and then multiply ``\sigma(C)`` onto ``Z`` from the right, i.e. the output of the attention layer is ``Z\sigma(C)``. So we perform the following mappings:

```math
Z \xrightarrow{\mathrm{correlations}} C(Z) =: C \xrightarrow{\sigma} \sigma(C) \xrightarrow{\text{right multiplication}} Z \sigma(C).
```


After the right multiplication the outpus is of the following form: 

```math 
    [\sum_{i=1}^Tp^{(1)}_iz^{(i)}, \ldots, \sum_{i=1}^Tp^{(T)}_iz^{(i)}],
```
for ``p^{(i)} = [\sigma(C)]_{\bullet{}i}``. What is *learned* during training are ``T`` different linear combinations of the input vectors, where the coefficients ``p^{(i)}_j`` in these linear combinations depend on the input ``Z`` nonlinearly. 

## `VolumePreservingAttention` in `GeometricMachineLearning`

The attention layer (and the activation function ``\sigma`` defined for it) in `GeometricMachineLearning` was specifically designed to apply it to data coming from physical systems that can be described through a divergence-free or a symplectic vector field. 
Traditionally the nonlinearity in the attention mechanism is a softmax[^3] (see [vaswani2017attention](@cite)) and the self-attention layer performs the following mapping: 

[^3]: The softmax acts on the matrix ``C`` in a vector-wise manner, i.e. it operates on each column of the input matrix ``C = [c^{(1)}, \ldots, c^{(T)}]``. The result is a sequence of probability vectors ``[p^{(1)}, \ldots, p^{(T)}]`` for which ``\sum_{i=1}^Tp^{(j)}_i=1\quad\forall{}j\in\{1,\dots,T\}.``

```math
Z := [z^{(1)}, \ldots, z^{(T)}] \mapsto Z\mathrm{softmax}(Z^TWZ).
```

The softmax activation acts vector-wise, i.e. if we supply it with a matrix ``C`` as input it returns: 

```math 
\mathrm{softmax}(C) = [\mathrm{softmax}(c_{\bullet{}1}), \ldots, \mathrm{softmax}(c_{\bullet{}T})].
```

The output of a softmax is a *probability vector* (also called *stochastic vector*) and the matrix ``[p^{(1)}, \ldots, p^{(T)}]``, where each column is a probability vector, is sometimes referred to as a *stochastic matrix* (see [jacobs1992discrete](@cite)). This attention mechanism finds application in *transformer neural networks*. The problem with this matrix from a geometric point of view is that all the columns are independent of each other and the nonlinear transformation could in theory produce a stochastic matrix for which all columns are identical and thus lead to a loss of information.

Besides the traditional attention mechanism `GeometricMachineLearning` hence also has a volume-preserving transformation that fulfills a similar role. There are two approaches implemented to realize similar transformations. Both of them however utilize the *Cayley transform* to produce orthonormal matrices ``\sigma(C)`` instead of stochastic matrices. 

### The Cayley transform 

The Cayley transform maps from skew-symmetric matrices to orthonormal matrices[^4]. It takes the form:

[^4]: A matrix ``A`` is skew-symmetric if ``A = -A^T`` and a matrix ``B`` is orthonormal if ``B^TB = \mathbb{I}``. The orthonormal matrices form a Lie group, i.e. the set of orthonormal matrices can be endowed with the structure of a differential manifold and this set also satisfies the group axioms. The corresponding Lie algebra are the skew-symmetric matrices and the Cayley transform is a so-called retraction in this case. For more details consult e.g. [hairer2006geometric](@cite) and [absil2008optimization](@cite).

```math 
\mathrm{Cayley}: A \mapsto (\mathbb{I} - A)(\mathbb{I} + A)^{-1}.
```

In order to use the Cayley transform as an activation function we further need a mapping from the input ``Z`` to a skew-symmetric matrix. This is realized in two ways in `GeometricMachineLearning` (these are the two approaches to which we referred to above).

### First approach: scalar products with a skew-symmetric weighting

For this the attention layer is modified in the following way: 

```math 
Z := [z^{(1)}, \ldots, z^{(T)}] \mapsto Z\sigma(Z^TAZ),
```
where ``\sigma(C)=\mathrm{Cayley}(C)`` and ``A`` is a skew-symmetric matrix that is learnable, i.e. the parameters of the attention layer are stored in ``A``. ``\mathrm{Cayley}(C) = (\mathbb(I) - A)(\mathbb{I} + A)^{-1}`` is the Cayley transform. 

### Second approach: scalar products with an arbitrary weighting



## How is structure preserved? 

To make the structure-preserving property more clear, consider that the transformer maps sequences of vectors to sequences of vectors, i.e. ``V\times\cdots\times{}V \ni [z^1, \ldots, z^T] \mapsto [\hat{z}^1, \ldots, \hat{z}^T]``. We can define a symplectic structure on ``V\times\cdots\times{}V`` by rearranging ``[z^1, \ldots, z^T]`` into a vector. We do this in the following way: 

```math
\tilde{Z} = \begin{pmatrix} q^{(1)}_1 \\ q^{(2)}_1 \\ \cdots \\ q^{(T)}_1 \\ q^{(1)}_2 \\ \cdots \\ q^{(T)}_d \\ p^{(1)}_1 \\ p^{(2)}_1 \\ \cdots \\ p^{(T)}_1 \\ p^{(1)}_2 \\ \cdots \\ p^{(T)}_d \end{pmatrix}.
```

Multiplying with the matrix ``\Lambda(Z)`` from the right onto ``[z^1, \ldots, z^T]`` corresponds to applying the sparse matrix 

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

Attention was used before, but always in connection with **recurrent neural networks** (see [luong2015effective](@cite) and [bahdanau2014neural](@cite)). 


## References 

```@bibliography
Pages = []
Canonical = false 

bahdanau2014neural
luong2015effective
```