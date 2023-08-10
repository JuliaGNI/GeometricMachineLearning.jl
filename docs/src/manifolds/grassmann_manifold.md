# Grassmann Manifold 

(The description of the Grassmann manifold is based on that of the Stiefel manifold, so this should be read first.)

An element of the Grassmann manifold $G(n,N)$ is a vector subspace $\subset\mathbb{R}^N$ of dimension $n$, and each such subspace can be represented by a full-rank matrix $A\in\mathbb{R}^{N\times{}n}$ and the full space takes the form $G(n,N) = \mathbb{R}^{N\times{}n}/\sim$ where the equivalence relation is $A\sim{}B \iff \exists{}C\in\mathbb{R}^{n\times{}n}\text{ s.t. }AC = B$. One can find a parametrization of the manifold the following way: Because the matrix $A$ has full rank, there have to be $n$ independent columns in it: $i_1, \ldots, i_n$. For simplicity assume that $i_1 = 1, i_2=2, \ldots, i_n=n$ and call the matrix made up by these columns $C$. Then the mapping to the coordinate chart is: $AC^{-1}$ and the last $N-n$ columns are the coordinates. 

The tangent space for this element can then be represented through matrices: 

```math
\begin{pmatrix}
    0 & \cdots & 0 \\
    \cdots & \cdots & \cdots \\ 
    0 & \cdots & 0 \\
    a_{11} & \cdots & a_{1n} \\
    \cdots & \cdots & \cdots \\ 
    a_{(N-n)1} & \cdots & a_{(N-n)n}
\end{pmatrix}.
```

The Grassmann manifold can also be seen as the Stiefel manifold modulo an equivalence class. This leads to the following (which is used for optimization):

```math
\mathfrak{g}^\mathrm{hor} = \mathfrak{g}^{\mathrm{hor},E} = \left\{\begin{pmatrix} 0 & -B^T \\ B & 0 \end{pmatrix}: \text{$B$ arbitrary}\right\}.
```
