# Multihead Attention

In order to arrive from the [attention layer](@ref "The Attention Layer") at the *multihead attention layer* we have to do a few modifications. Here note that these neural networks were originally developed for natural language processing (NLP) tasks and the terminology used here bears some resemblance to that field. 
The input to a multihead attention layer typicaly comprises three components:

1. Values ``V\in\mathbb{R}^{N\times{}T}``: a matrix whose columns are *value vectors*, 
2. Queries ``Q\in\mathbb{R}^{N\times{}T}``: a matrix whose columns are *query vectors*, 
3. Keys ``K\in\mathbb{R}^{N\times{}T}``: a matrix whose columns are *key vectors*.

Regular attention performs the following operation[^1]: 

[^1]: The division by ``\sqrt{N}`` is optional here and sometimes left out.

```math
\mathrm{Attention}(Q,K,V) = V\mathrm{softmax}\left(\frac{K^TQ}{\sqrt{N}}\right),
```

where ``N`` is the dimension of the vectors in ``V``, ``Q`` and ``K``. The softmax activation function here acts column-wise:

```math
\mathrm{softmax}:\mathbb{R}^{T}\to\mathbb{R}^T \text{ with $[\mathrm{softmax}(v)]_i = e^{v_i}/\left(\sum_{j=1}e^{v_j}\right)$.}
``` 
The ``K^TQ`` term is a similarity matrix between the queries and the vectors. 

The transformer contains a *self-attention mechanism*, i.e. takes an input ``X`` and then transforms it linearly to ``V``, ``Q`` and ``K`` via ``V = P^VX``, ``Q = P^QX`` and ``K = P^KX``. What distinguishes the multihead attention layer from the singlehead attention layer is that there is not just one ``P^V``, ``P^Q`` and ``P^K``, but there are several: one for each *head* of the multihead attention layer. After computing the individual values, queries and vectors, and after applying the softmax, the outputs are then concatenated together in order to obtain again an array that is of the same size as the input array:

```@example 
Main.include_graphics("../tikz/mha"; caption = raw"A representation of a multihead attention layer with three heads. ") # hide
```

Written as an equation we get:

```math
\mathrm{MultiHeadAttention}(Z) = \begin{pmatrix} \mathrm{Attention}(P^Q_1Z, P^K_1Z, P^V_1Z) \\ \mathrm{Attention}(P^Q_2Z, P^K_2Z, P^V_2Z) \\ \cdots \\ \mathrm{Attention}(P^Q_{\mathtt{n\_heads}}Z, P^K_{\mathtt{n\_heads}}Z, P^V_{\mathtt{n\_heads}}Z) \end{pmatrix},
```

where ``P^{(\cdot)}_i\in\mathbb{R}^{N\times(N\div\mathtt{n\_heads})}`` for ``Z\in\mathbb{R}^{N\times{}T}.`` Note that we implicitly require that ``N`` is divisible by ``\mathtt{n\_heads}`` here.

Here the various ``P`` matrices can be interpreted as being projections onto lower-dimensional subspaces, hence the designation by the letter ``P``. The columns of the projection matrices span smaller spaces that should *capture features in the input data*. We will show [in an example](@ref "MNIST Tutorial") how training of a neural network can benefit from putting the ``P^{(\cdot)}_i`` matrices on the Stiefel manifold.   

```@eval
Main.remark(raw"The `MultiHeadAttention` implemented in `GeometricMachineLearning` has an optional keyword `add_connection`. If this is set to `true` then the output of the `MultiHeadAttention` layer is:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\mathrm{MultiHeadAttention}(Z) = Z + \begin{pmatrix} \mathrm{Attention}(P^Q_1Z, P^K_1Z, P^V_1Z) \\ \mathrm{Attention}(P^Q_2Z, P^K_2Z, P^V_2Z) \\ \cdots \\ \mathrm{Attention}(P^Q_{\mathtt{n\_heads}}Z, P^K_{\mathtt{n\_heads}}Z, P^V_{\mathtt{n\_heads}}Z) \end{pmatrix},
" * Main.indentation * raw"```
" * Main.indentation * raw"so we add the input again to the output.")
```

## Computing Correlations in the Multihead-Attention Layer

The attention mechanism describes a reweighting of the "values" ``V_i`` based on correlations between the "keys" ``K_i`` and the "queries" ``Q_i``. First note the structure of these matrices: they are all a collection of ``T`` ``(N\div\mathtt{n\_heads})``-dimensional vectors, i.e. ``V_i=[v_i^{(1)}, \ldots, v_i^{(T)}], K_i=[k_i^{(1)}, \ldots, k_i^{(T)}], Q_i=[q_i^{(1)}, \ldots, q_i^{(T)}]`` with ``i = 1, \ldots, \mathtt{n\_heads}``. Those vectors have been obtained by applying the respective projection matrices onto the original input.

When performing the *reweighting* of the columns of ``V_i`` we first compute the correlations between the vectors in ``K_i`` and in ``Q_i`` and store the results in a *correlation matrix* ``C_i``: 

```math
    [C_i]_{mn} = \left(k_i^{(m)}\right)^Tq_i^{(n)}.
```

The columns of this correlation matrix are than rescaled with a softmax function, obtaining a matrix of *probability vectors*[^2] ``\mathcal{P}_i``:

[^2]: Also called a *stochastic matrix*.

```math
    [\mathcal{P}_i]_{\bullet{}n} = \mathrm{softmax}\left(\frac{[C_i]_{\bullet{}n}}{\sqrt{N\div\mathtt{n\_heads}}}\right).
```

Finally the matrix ``\mathcal{P}_i`` is multiplied onto ``V_i`` from the right, resulting in ``T`` convex combinations of the ``T`` vectors ``v_i^{(m)}`` with ``m=1,\ldots,T``:

```math
    V_i\mathcal{P}_i = \left[\sum_{m=1}^{T}[\mathcal{P}_i]_{m,1}v_i^{(m)}, \ldots, \sum_{m=1}^{T}[\mathcal{P}_i]_{m,T}v_i^{(m)}\right].
```

With this we can now give a better interpretation of what the projection matrices ``W_i^V``, ``W_i^K`` and ``W_i^Q`` should do: they map the original data to lower-dimensional subspaces. We then compute correlations between the representation in the $K$ and in the ``Q`` basis and use this correlation to perform a convex reweighting of the vectors in the $V$ basis. These reweighted *values* are then fed into a standard feedforward neural network as is further explained in the [section on the standard transformer](@ref "Standard Transformer").

Because the main task of the ``W_i^V``, ``W_i^K`` and ``W_i^Q`` matrices here is for them to find bases, it makes sense to constrain them onto the Stiefel manifold; they do not and should not have the maximum possible generality.

## Library Functions 

```@docs
MultiHeadAttention
```

## References 

```@bibliography
Pages = []
Canonical = false

vaswani2017attention
```