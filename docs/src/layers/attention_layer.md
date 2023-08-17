# The Attention Layer

The attention layer (and the *orthonormal activation* function defined for it) was specifically designed to generalize transformers to symplectic data. 
Usually a self-attention layer takes the following form: 

$$
Z := [z_1, \ldots, z_T] \mapsto Z\mathrm{softmax}((P^QZ)^T(P^KZ)),
$$
where we left out the linear mapping onto the values $P^V$. 

The idea behind is that we can perform a non-linear re-weighting of the columns of $Z$ by multiplying with a $Z$-dependent matrix from the right and therefore take the sequential nature of the data into account (which is not possible with normal neural networks). After the attention step the transformer applies a simple ResNet from the left.

What the softmax does is a vector-wise operation, i.e. it operates on each column of an input matrix $A = [a_1, \ldots, a_T]$. The result is a sequence of probability vectors $[p^1, \ldots, p^T]$ for which $\sum_{i=1}^Tp^j_i = 1\quad\forall{}j\in\{1,\dots,T\}$. 

What we want to construct is a symplectic transformation that is *transformer-like*. For this we modify the attention layer the following way: 

$$
Z := [z_1, \ldots, z_T] \mapsto Z\sigma((P^QZ)^T(P^KZ)),
$$
where $\sigma(A) = \exp(\mathtt{upper\_triangular\_asymmetrize}(A))$ and 

$$
[\mathtt{upper\_triangular\_asymmetrize}(A)]_{ij} = \begin{cases} a_{ij} & \text{if $i<j$}  \\ -a_{ji} & \text{if $i>j$} \\ 0 & \text{else.}\end{cases}
$$

This has as a consequence that the matrix $\sigma((P^QZ)^T(P^KZ))$ is orthonormal. 