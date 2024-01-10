# Grassmann Manifold 

(The description of the Grassmann manifold is based on that of the [Stiefel manifold](stiefel_manifold.md), so this should be read first.)

An element of the Grassmann manifold $G(n,N)$ is a vector subspace $\subset\mathbb{R}^N$ of dimension $n$. Each such subspace (i.e. element of the Grassmann manifold) can be represented by a full-rank matrix $A\in\mathbb{R}^{N\times{}n}$ and we identify two elements with the following equivalence relation: 

```math
A_1 \sim A_2 \iff \exists{}C\in\mathbb{R}^{n\times{}n}\text{ s.t. }A_1C = A_2
```

We can also define the Grassmann manifold[^1] based on the Stiefel manifold since elements of the Stiefel manifold are already full-rank matrices. In this case we have the following equivalence relation (for ``Y_1, Y_2\in{}St(n,N)``): 

```math
Y_1 \sim Y_2 \iff \exists{}C\in{}O(n)\text{ s.t. }Y_1C = Y_2.
```

[^1]: One can find a parametrization of the manifold the following way: Because the matrix ``Y`` has full rank, there have to be ``n`` independent columns in it: ``i_1, \ldots, i_n``. For simplicity assume that ``i_1 = 1, i_2=2, \ldots, i_n=n`` and call the matrix made up by these columns ``C``. Then the mapping to the coordinate chart is: ``YC^{-1}`` and the last ``N-n`` columns are the coordinates. 

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

## The Riemannian Gradient

For matrix manifolds (like the Stiefel manifold), the Riemannian gradient of a function can be easily determined computationally:

The Euclidean gradient of a function $L$ is equivalent to an element of the cotangent space $T^*_Y\mathcal{M}$ via: 
```math
\langle\nabla{}L,\cdot\rangle:T_Y\mathcal{M} \to \mathbb{R}, \Delta \mapsto \sum_{ij}[\nabla{}L]_{ij}[\Delta]_{ij} = \mathrm{Tr}(\nabla{}L^T\Delta).
```

We can then utilize the Riemannian metric on $\mathcal{M}$ to map the element from the cotangent space (i.e. $\nabla{}L$) to the tangent space. This element is called $\mathrm{grad}_{(\cdot)}L$ here. Explicitly, it is given by: 

```math
    \mathrm{grad}_YL = \nabla_YL - Y(\nabla_YL)^TY
```

### `rgrad`

What was referred to as $\nabla{}L$ before can in practice be obtained with an AD routine. We then use the function `rgrad` to map this *Euclidean gradient* to $\in{}T_YSt(n,N)$. This mapping has the property: 

```math 
\mathrm{Tr}((\nabla{}L)^T\Delta) = g_Y(\mathtt{rgrad}(Y, \nabla{}L), \Delta) \forall\Delta\in{}T_YSt(n,N)
```

 and $g$ is the Riemannian metric.