# Stiefel manifold 

The Stiefel manifold $St(n, N)$ is the space (a [homogeneous space](homogeneous_spaces.md)) of all orthonormal frames in $\mathbb{R}^{N\times{}n}$, i.e. matrices $Y\in\mathbb{R}^{N\times{}n}$ s.t. $Y^TY = \mathbb{I}_n$. It can also be seen as the special orthonormal group $SO(N)$ modulo an equivalence relation: $A\sim{}B\iff{}AE = BE$ for 

```math
E = \begin{bmatrix}
\mathbb{I}_n \\ 
\mathbb{O}
\end{bmatrix}\in\mathcal{M},
```
which is the canonical element of the Stiefel manifold. In words: the first $n$ columns of $A$ and $B$ are the same.

The tangent space to the element $Y\in{}St(n,N)$ can easily be determined: 
```math
T_YSt(n,N)=\{\Delta:\Delta^TY + Y^T\Delta = 0\}.
```

The Lie algebra of $SO(N)$ is $\mathfrak{so}(N):=\{V\in\mathbb{R}^{N\times{}N}:V^T + V = 0\}$ and the canonical metric associated with it is simply $(V_1,V_2)\mapsto\frac{1}{2}\mathrm{Tr}(V_1^TV_2)$.


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