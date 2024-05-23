# Homogeneous Spaces 

*Homogeneous spaces* are very important in `GeometricMachineLearning` as we can generalize existing neural network optimizers from vector spaces to such homegenous spaces. They are intricately linked to the notion of a *Lie Group* and its *Lie Algebra*[^1].

[^1]: Recall that a Lie group is a manifold that also has group structure. We say that a Lie group ``G`` *acts* on a manifold ``\mathcal{M}`` if there is a map ``G\times\mathcal{M} \to \mathcal{M}`` such that ``(ab)x = a(bx)`` for ``a,b\in{}G`` and ``x\in\mathcal{M}``. For us the Lie algebra belonging to a Lie group, denoted by ``\mathfrak{g}``, is the tangent space to the identity element ``T_\mathbb{I}G``. 

```@eval
Main.definition(raw"A **homogeneous space** is a manifold ``\mathcal{M}`` on which a Lie group ``G`` acts transitively, i.e.
" * Main.indentation * raw" ```math
" * Main.indentation * raw"\forall X,Y\in\mathcal{M} \exists{}A\in{}G\text{ s.t. }AX = Y.
" * Main.indentation * raw"```
")

Now fix a distinct element ``E\in\mathcal{M}``. We can also establish an isomorphism between ``\mathcal{M}`` and the quotient space ``G/\sim`` with the equivalence relation: 
```math
A_1 \sim A_2 \iff A_1E = A_2E.
```
Note that this is independent of the chosen ``E``.

The tangent spaces of ``\mathcal{M}`` are of the form ``T_Y\mathcal{M} = \mathfrak{g}\cdot{}Y``, i.e. can be fully described through its Lie algebra. 
Based on this we can perform a splitting of ``\mathfrak{g}`` into two parts:
1. The *vertical component* ``\mathfrak{g}^{\mathrm{ver},Y}`` is the kernel of the map ``\mathfrak{g}\to{}T_Y\mathcal{M}, V \mapsto VY``, i.e. ``\mathfrak{g}^{\mathrm{ver},Y} = \{V\in\mathfrak{g}:VY = 0\}.``
2. The *horizontal component* ``\mathfrak{g}^{\mathrm{hor},Y}`` is the orthogonal complement of ``\mathfrak{g}^{\mathrm{ver},Y}`` in ``\mathfrak{g}``. It is isomorphic to ``T_Y\mathcal{M}``.

We will refer to the mapping from ``T_Y\mathcal{M}$ to $\mathfrak{g}^{\mathrm{hor}, Y}`` by ``\Omega``. We will give explicit examples of ``\Omega`` below. If we have now defined a metric ``\langle\cdot,\cdot\rangle`` on ``\mathfrak{g}``, then this induces a Riemannian metric on ``\mathcal{M}``:
```math
g_Y(\Delta_1, \Delta_2) = \langle\Omega(Y,\Delta_1),\Omega(Y,\Delta_2)\rangle\text{ for $\Delta_1,\Delta_2\in{}T_Y\mathcal{M}$.}
```

Two examples of homogeneous spaces implemented in `GeometricMachineLearning` are the [Stiefel](@ref "The Stiefel Manifold") and the [Grassmann](@ref "The Grassmann Manifold") manifold. The Lie group ``SO(N)`` acts transitively on both of these manifolds, i.e. turns them into homogeneous spaces.

# The Stiefel Manifold 

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

 and ``g`` is the Riemannian metric.

# The Grassmann Manifold 

(The description of the Grassmann manifold is based on that of the [Stiefel manifold](stiefel_manifold.md), so this should be read first.)

An element of the Grassmann manifold $G(n,N)$ is a vector subspace $\subset\mathbb{R}^N$ of dimension $n$. Each such subspace (i.e. element of the Grassmann manifold) can be represented by a full-rank matrix $A\in\mathbb{R}^{N\times{}n}$ and we identify two elements with the following equivalence relation: 

```math
A_1 \sim A_2 \iff \exists{}C\in\mathbb{R}^{n\times{}n}\text{ s.t. }A_1C = A_2.
```

The resulting manifold is of dimension ``n(N-n)``. One can find a parametrization of the manifold the following way: Because the matrix ``Y`` has full rank, there have to be ``n`` independent columns in it: ``i_1, \ldots, i_n``. For simplicity assume that ``i_1 = 1, i_2=2, \ldots, i_n=n`` and call the matrix made up by these columns ``C``. Then the mapping to the coordinate chart is: ``YC^{-1}`` and the last ``N-n`` columns are the coordinates.

We can also define the Grassmann manifold based on the Stiefel manifold since elements of the Stiefel manifold are already full-rank matrices. In this case we have the following equivalence relation (for ``Y_1, Y_2\in{}St(n,N)``): 

```math
Y_1 \sim Y_2 \iff \exists{}C\in{}O(n)\text{ s.t. }Y_1C = Y_2.
```

## The Riemannian Gradient

Obtaining the Riemannian Gradient for the Grassmann manifold is slightly more difficult than it is in the case of the Stiefel manifold. Since the Grassmann manifold can be obtained from the Stiefel manifold through an equivalence relation however, we can use this as a starting point. In a first step we identify charts on the Grassmann manifold to make dealing with it easier. For this consider the following open cover of the Grassmann manifold (also see [absil2004riemannian](@cite)): 

```math
\{\mathcal{U}_W\}_{W\in{}St(n, N)} \quad\text{where}\quad \mathcal{U}_W = \{\mathrm{span}(Y):\mathrm{det}(W^TY)\neq0\}.
```

We can find a canonical bijective mapping from the set ``\mathcal{U}_W`` to the set ``\mathcal{S}_W := \{Y\in\mathbb{R}^{N\times{}n}:W^TY=\mathbb{I}_n\}``:

```math
\sigma_W: \mathcal{U}_W \to \mathcal{S}_W,\, \mathcal{Y}=\mathrm{span}(Y)\mapsto{}Y(W^TY)^{-1} =: \hat{Y}.
```

That ``\sigma_W`` is well-defined is easy to see: Consider ``YC`` with ``C\in\mathbb{R}^{n\times{}n}`` non-singular. Then ``YC(W^TYC)^{-1}=Y(W^TY)^{-1} = \hat{Y}``. With this isomorphism we can also find a representation of elements of the tangent space:

```math
T_\mathcal{Y}\sigma_W: T_\mathcal{Y}Gr(n,N)\to{}T_{\hat{Y}}\mathcal{S}_W,\, \xi \mapsto (\xi_{\diamond{}Y} -\hat{Y}(W^T\xi_{\diamond{}Y}))(W^TY)^{-1}.
```

``\xi_{\diamond{}Y}`` is the representation of ``\xi\in{}T_\mathcal{Y}Gr(n,N)`` for the point ``Y\in{}St(n,N)``, i.e. ``T_Y\pi(\xi_{\diamond{}Y}) = \xi``; because the map ``\sigma_W`` does not care about the representation of ``\mathrm{span}(Y)`` we can perform the variations in ``St(n,N)``[^1]:

[^1]: I.e. ``Y(t)\in{}St(n,N)`` for ``t\in(-\varepsilon,\varepsilon)``. We also set ``Y(0) = Y``.

```math
\frac{d}{dt}Y(t)(W^TY(t))^{-1} = (\dot{Y}(0) - Y(W^TY)^{-1}W^T\dot{Y}(0))(W^TY)^{-1},
```

where ``\dot{Y}(0)\in{}T_YSt(n,N)``. Also note that the representation of ``\xi`` in ``T_YSt(n,N)`` is not unique in general, but ``T_\mathcal{Y}\sigma_W`` is still well-defined. To see this consider two curves ``Y(t)`` and ``\bar{Y}(t)`` for which we have ``Y(0) = \bar{Y}(0) = Y`` and further ``T\pi(\dot{Y}(0)) = T\pi(\dot{bar{Y}}(0))``. This is equivalent to being able to find a ``C(\cdot):(-\varepsilon,\varepsilon)\to{}O(n)`` for which ``C(0)=\mathbb{I}(0)`` s.t. ``\bar{Y}(t) = Y(t)C(t)``. We thus have ``\dot{\bar{Y}}(0) = \dot{Y}(0) + Y\dot{C}(0)`` and if we replace ``\xi_{\diamond{}Y}`` above with the second term in the expression we get: ``Y\dot{C}(0) - \hat{Y}W^T(Y\dot{C}(0)) = 0``. The parametrization of ``T_\mathcal{Y}Gr(n,N)`` with ``T_\mathcal{Y}\sigma_W`` is thus independent of the choice of ``\dot{C}(0)`` and hence of ``\xi_{\diamond{}Y}`` and is therefore well-defined.


Further note that we have ``T_\mathcal{Y}\mathcal{U}_W = T_\mathcal{Y}Gr(n,N)`` because ``\mathcal{U}_W`` is an open subset of ``Gr(n,N)``. We thus can identify the tangent space ``T_\mathcal{Y}Gr(n,N)`` with the following set (where we again have ``\hat{Y}=Y(W^TY)^{-1}``):

```math
T_{\hat{Y}}\mathcal{S}_W = \{(\Delta - Y(W^TY)^{-1}W^T\Delta)(W^T\Delta)^{-1}: Y\in{}St(n,N)\text{ s.t. }\mathrm{span}(Y)=\mathcal{Y}\text{ and }\Delta\in{}T_YSt(n,N)\}.
```

If we now further take ``W=Y``[^2] then we get the identification: 

[^2]: We can pick any element ``W`` to construct the charts for a neighborhood around the point ``\mathcal{Y}\in{}Gr(n,N)`` as long as we have ``\mathrm{det}(W^TY)\neq0`` for ``\mathrm{span}(Y)=\mathcal{Y}``. 

```math
T_\mathcal{Y}Gr(n,N) \equiv \{\Delta - YY^T\Delta: Y\in{}St(n,N)\text{ s.t. }\mathrm{span}(Y)=\mathcal{Y}\text{ and }\Delta\in{}T_YSt(n,N)\},
```
which is very easy to handle computationally (we simply store and change the matrix ``Y`` that represents an element of the Grassmann manifold). The Riemannian gradient is then 

```math
\mathrm{grad}_\mathcal{Y}^{Gr}L = \mathrm{grad}_Y^{St}L - YY^T\mathrm{grad}_Y^{St}L = \nabla_Y{}L - YY^T\nabla_YL,
```
where ``\nabla_Y{}L`` again is the Euclidean gradient as in the [Stiefel manifold](stiefel_manifold.md) case.

 ## Library Functions 

```@docs; canonical=false
StiefelManifold
GeometricMachineLearning.rgrad(::StiefelManifold, ::AbstractMatrix)
GeometricMachineLearning.metric(::StiefelManifold, ::AbstractMatrix, ::AbstractMatrix)
```

## References 

```@bibliography
Pages = []
Canonical = false

frankel2011geometry
bendokat2021real
```