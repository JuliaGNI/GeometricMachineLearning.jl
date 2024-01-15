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

Obtaining the Riemannian Gradient for the Grassmann manifold is slightly more difficult than it is in the case of the Stiefel manifold. Since the Grassmann manifold can be obtained from the Stiefel manifold through an equivalence relation however, we can use this as a starting point. In a first step we identify charts on the Grassmann manifold to make dealing with it easier. For this consider the following open cover of the Grassmann manifold (also see [absil2004riemannian](@cite)): 

```math
\{\mathcal{U}_W\}_{W\in{}St(n, N)} \text{ where } \mathcal{U}_W = \{\mathrm{span}(Y):\mathrm{det}(W^TY)\neq0\}.
```

We can find a canonical bijective mapping from the set ``\mathcal{U}_W`` to the set ``\mathcal{S}_W := \{Y\in\mathbb{R}^{N\times{}n}:W^TY=\mathbb{I}_n\}``:

```math
\sigma_W: \mathcal{U}_W \to \mathcal{S}_W,\, \mathcal{Y}=\mathrm{span}(Y)\mapsto{}Y(W^TY)^{-1} =: \hat{Y}.
```

That ``\sigma_W`` is well-defined is easy to see: Consider ``YC`` with ``C\in\mathbb{R}^{n\times{}n}`` non-singular. Then ``YC(W^TYC)^{-1}=Y(W^TY)^{-1}``. With this isomorphism we can also find a representation of elements of the tangent space:

```math
T_\mathcal{Y}\sigma_W: T_\mathcal{Y}Gr(n,N)\to{}T_\hat{Y}\mathcal{S}_W,\, \xi \mapsto (\xi_{\diamond{}Y} -\hat{Y}W^T\xi_{\diamond{}Y})(W^TY)^{-1}.
```

``\xi_{\diamond{}Y}`` is the representation of ``\xi\in{}T_\mathcal{Y}Gr(n,N)`` for the point ``Y\in{}St(n,N)``; because the map ``\sigma_W`` does not care about the representation of ``\mathrm{span}(Y)`` we can perform the variations in ``St(n,N)``[^2]:

[^2]: I.e. ``Y(t)\in{}St(n,N)`` for ``t\in(-\varepsilon,\varepsilon)``. We also set ``Y(0) = Y``.

```math
\frac{d}{dt}Y(t)(W^TY(t))^{-1} = (\dot{Y}(0) - Y(W^TY)^{-1}W^T\dot{Y}(0))(W^TY)^{-1},
```

where ``\dot{Y}(0)\in{}T_YSt(n,N)``. Further note that we have ``T_\mathcal{Y}\mathcal{U}_W = T_\mathcal{Y}Gr(n,N)`` because ``\mathcal{U}_W`` is an open subset of ``Gr(n,N)``. We thus can identify the tangent space ``T_\mathcal{Y}Gr(n,N)`` with the following set:

```math
T_{\hat{Y}}\mathcal{S}_W = \{(\Delta - Y(W^TY)^{-1}W^T\Delta)(W^T\Delta)^{-1}: Y\in{}St(n,N)\text{ s.t. }\mathrm{span}(Y)=\mathcal{Y}\text{ and }\Delta\in{}T_YSt(n,N)\}.
```

If we now further take ``W=Y`` for our distinct element on whose basis we construct the charts, then we get the identification: 

```math
T_\mathcal{Y}Gr(n,N) = \{\Delta - YY^T\Delta: Y\in{}St(n,N)\text{ s.t. }\mathrm{span}(Y)=\mathcal{Y}\text{ and }\Delta\in{}T_YSt(n,N)\},
```
which is very easy to handle computationally (we simply store and change the matrix ``Y`` that represents an element of the Grassmann manifold). The Riemannian gradient is then 

```math
\mathrm{grad}_\mathcal{Y}^{Gr}L = \mathrm{grad}_Y^{St}L - YY^T\mathrm{grad}_Y^{St}L = \nabla_Y{}L - YY^T\nabla_YL,
```
where ``\nabla_Y{}L`` again is the Euclidean gradient as in the [Stiefel manifold](stiefel_manifold.md) case.