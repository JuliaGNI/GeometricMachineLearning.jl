# The Attention Layer

The *attention* mechanism was originally applied for image and natural language processing (NLP) tasks. In (Bahdanau et al, 2014) ``additive'' attention is used: 

```math
(z_q, z_k) \mapsto v^T\sigma(Wz_q + Uz_k).
```

However ``multiplicative'' attention is more straightforward to interpret and cheaper to handle computationally: 

```math
(z_q, z_k) \mapsto z_q^TWz_k.
```

Regardless of the type of attention used, they all try to compute correlations among input sequences on whose basis further neural network-based computation is performed. So given two input sequences $(z_q^{(1)}, \ldots, z_q^{(T)})$ and $(z_k^{(1)}, \ldots, z_k^{(T)})$, various attention mechanisms always return an output $C\in\mathbb{R}^{T\times{}T}$ with entries $[C]_{ij} = \mathtt{attention}(z_q^{(i)}, z_k^{(j)}$.

# Self Attention 




## Attention in `GeometricMachineLearning`

The attention layer (and the *orthonormal activation* function defined for it) in `GeometricMachineLearning` was specifically designed to generalize transformers to symplectic data. 
Usually a self-attention layer takes the following form: 

```math
Z := [z^{(1)}, \ldots, z^{(T)}] \mapsto Z\mathrm{softmax}((P^QZ)^T(P^KZ)),
```
where we left out the linear mapping onto the values $P^V$. 

The idea behind is that we can perform a non-linear re-weighting of the columns of $Z$ by multiplying with a $Z$-dependent matrix from the right and therefore take the sequential nature of the data into account (which is not possible with normal neural networks). After the attention step the transformer applies a simple ResNet from the left.

What the softmax does is a vector-wise operation, i.e. it operates on each column of an input matrix $A = [a_1, \ldots, a_T]$. The result is a sequence of probability vectors $[p^{(1)}, \ldots, p^{(T)}]$ for which 

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
- Luong M T, Pham H, Manning C D. Effective approaches to attention-based neural machine translation[J]. arXiv preprint arXiv:1508.04025, 2015.
- Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.