# The horizontal component of the Lie algebra ``\mathfrak{g}`` for the Grassmann manifold

## Tangent space to the element ``\mathcal{E}``

Consider the tangent space to the distinct element ``\mathcal{E}=\mathrm{span}(E)\in{}Gr(n,N)``, where ``E`` is again:

```math
E = \begin{bmatrix}
\mathbb{I}_n \\ 
\mathbb{O}
\end{bmatrix}.
```


The tangent tangent space ``T_\mathcal{E}Gr(n,N)`` can be represented through matrices: 

```math
\begin{pmatrix}
    0 & \cdots & 0 \\
    \cdots & \cdots & \cdots \\ 
    0 & \cdots & 0 \\
    a_{11} & \cdots & a_{1n} \\
    \cdots & \cdots & \cdots \\ 
    a_{(N-n)1} & \cdots & a_{(N-n)n}
\end{pmatrix},
```

where we have used the identification ``T_\mathcal{E}Gr(n,N)\to{}T_E\mathcal{S}_E`` that was discussed in the [section on the Grassmann manifold](../manifolds/grassmann_manifold.md).  The Grassmann manifold can also be seen as the Stiefel manifold modulo an equivalence class. This leads to the following (which is used for optimization):

```math
\mathfrak{g}^\mathrm{hor} = \mathfrak{g}^{\mathrm{hor},\mathcal{E}} = \left\{\begin{pmatrix} 0 & -B^T \\ B & 0 \end{pmatrix}: \text{$B$ arbitrary}\right\}.
```

This is equivalent to the horizontal component of ``\mathfrak{g}`` for the [Stiefel manifold](stiefel_lie_alg_horizontal.md) for the case when ``A`` is zero. This is a reflection of the rotational invariance of the Grassmann manifold: the skew-symmetric matrices ``A`` are connected to the group of rotations ``O(n)`` which is factored out in the Grassmann manifold ``Gr(n,N)\simeq{}St(n,N)/O(n)``.