# Multihead Attention

In order to arrive from the [attention layer](@ref "The Attention Layer") at the **multihead attention layer** we have to do a few modifications: 

Note that these neural networks were originally developed for natural language processing (NLP) tasks and the terminology used here bears some resemblance to that field. 
The input to a multihead attention layer typicaly comprises three components:

1. Values $V\in\mathbb{R}^{n\times{}T}$: a matrix whose columns are **value vectors**, 
2. Queries $Q\in\mathbb{R}^{n\times{}T}$: a matrix whose columns are **query vectors**, 
3. Keys $K\in\mathbb{R}^{n\times{}T}$: a matrix whose columns are **key vectors**.

Regular attention performs the following operation: 

```math
\mathrm{Attention}(Q,K,V) = V\mathrm{softmax}(\frac{K^TQ}{\sqrt{n}}),
```

where $n$ is the dimension of the vectors in $V$, $Q$ and $K$. The softmax activation function here acts column-wise, so it can be seen as a transformation $\mathrm{softmax}:\mathbb{R}^{T}\to\mathbb{R}^T$ with $[\mathrm{softmax}(v)]_i = e^{v_i}/\left(\sum_{j=1}e^{v_j}\right)$. The $K^TQ$ term is a similarity matrix between the queries and the vectors. 

The transformer contains a **self-attention mechanism**, i.e. takes an input $X$ and then transforms it linearly to $V$, $Q$ and $K$, i.e. $V = P^VX$, $Q = P^QX$ and $K = P^KX$. What distinguishes the multihead attention layer from the singlehead attention layer, is that there is not just one $P^V$, $P^Q$ and $P^K$, but there are several: one for each **head** of the multihead attention layer. After computing the individual values, queries and vectors, and after applying the softmax, the outputs are then concatenated together in order to obtain again an array that is of the same size as the input array:

```@example 
Main.include_graphics("../tikz/mha") # hide
```

Here the various $P$ matrices can be interpreted as being projections onto lower-dimensional subspaces, hence the designation by the letter $P$. Because of this interpretation as projection matrices onto smaller spaces that should **capture features in the input data** it makes sense to constrain these elements to be part of the Stiefel manifold.   

## Computing Correlations in the Multihead-Attention Layer

The [attention mechanism](attention_layer.md) describes a reweighting of the "values" $V_i$ based on correlations between the "keys" $K_i$ and the "queries" $Q_i$. First note the structure of these matrices: they are all a collection of $T$ vectors $(N\div\mathtt{n\_heads})$-dimensional vectors, i.e. $V_i=[v_i^{(1)}, \ldots, v_i^{(T)}], K_i=[k_i^{(1)}, \ldots, k_i^{(T)}], Q_i=[q_i^{(1)}, \ldots, q_i^{(T)}]$ . Those vectors have been obtained by applying the respective projection matrices onto the original input $I_i\in\mathbb{R}^{N\times{}T}$.

When performing the *reweighting* of the columns of $V_i$ we first compute the correlations between the vectors in $K_i$ and in $Q_i$ and store the results in a *correlation matrix* $C_i$: 

```math
    [C_i]_{mn} = \left(k_i^{(m)}\right)^Tq_i^{(n)}.
```

The columns of this correlation matrix are than rescaled with a softmax function, obtaining a matrix of *probability vectors* $\mathcal{P}_i$:

```math
    [\mathcal{P}_i]_{\bullet{}n} = \mathrm{softmax}([C_i]_{\bullet{}n}).
```

Finally the matrix $\mathcal{P}_i$ is multiplied onto $V_i$ from the right, resulting in 16 convex combinations of the 16 vectors $v_i^{(m)}$ with $m=1,\ldots,T$:

```math
    V_i\mathcal{P}_i = \left[\sum_{m=1}^{16}[\mathcal{P}_i]_{m,1}v_i^{(m)}, \ldots, \sum_{m=1}^{T}[\mathcal{P}_i]_{m,T}v_i^{(m)}\right].
```

With this we can now give a better interpretation of what the projection matrices $W_i^V$, $W_i^K$ and $W_i^Q$ should do: they map the original data to lower-dimensional subspaces. We then compute correlations between the representation in the $K$ and in the $Q$ basis and use this correlation to perform a convex reweighting of the vectors in the $V$ basis. These reweighted *values* are then fed into a standard feedforward neural network.

Because the main task of the $W_i^V$, $W_i^K$ and $W_i^Q$ matrices here is for them to find bases, it makes sense to constrain them onto the Stiefel manifold; they do not and should not have the maximum possible generality.

## Library Functions 

```@docs; canonical=false
MultiHeadAttention
```

## References 

```@bibliography
Pages = []
Canonical = false

vaswani2017attention
```