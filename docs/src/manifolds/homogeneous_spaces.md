# Homogeneous Spaces 

*Homogeneous spaces* are very important in `GeometricMachineLearning` as we can generalize existing neural network optimizers from vector spaces to such homogenous spaces. They are intricately linked to the notion of a *Lie Group* and its *Lie Algebra*[^1].

[^1]: Recall that a Lie group is a manifold that also has group structure. We say that a Lie group ``G`` *acts* on a manifold ``\mathcal{M}`` if there is a map ``G\times\mathcal{M} \to \mathcal{M}`` such that ``(ab)x = a(bx)`` for ``a,b\in{}G`` and ``x\in\mathcal{M}``. For us the Lie algebra belonging to a Lie group, denoted by ``\mathfrak{g}``, is the tangent space to the identity element ``T_\mathbb{I}G``. 

```@eval
Main.definition(raw"A **homogeneous space** is a manifold ``\mathcal{M}`` on which a Lie group ``G`` acts transitively, i.e.
" * Main.indentation * raw" ```math
" * Main.indentation * raw"\forall X,Y\in\mathcal{M} \quad \exists{}A\in{}G\text{ s.t. }AX = Y.
" * Main.indentation * raw"```
")
```

Now fix a distinct element ``E\in\mathcal{M}``; we will refer to this as the *canonical element* or [`StiefelProjection`](@ref). We can also establish an isomorphism between ``\mathcal{M}`` and the quotient space ``G/\sim`` with the equivalence relation: 
```math
A_1 \sim A_2 \iff A_1E = A_2E.
```
Note that this is independent of the chosen ``E``.

The tangent spaces of ``\mathcal{M}`` are of the form ``T_Y\mathcal{M} = \mathfrak{g}\cdot{}Y``, i.e. can be fully described through its Lie algebra. 
Based on this we can perform a splitting of ``\mathfrak{g}`` into two parts:

```@eval
Main.definition(raw"A **splitting of the Lie algebra** ``\mathfrak{g}`` at an element of a homogeneous space ``Y`` is a decomposition into a **vertical** and a **horizontal** component, denoted by ``\mathfrak{g} = \mathfrak{g}^{\mathrm{ver},Y} \oplus \mathfrak{g}^{\mathrm{hor},Y}`` such that
" * Main.indentation * raw"1. The *vertical component* ``\mathfrak{g}^{\mathrm{ver},Y}`` is the kernel of the map ``\mathfrak{g}\to{}T_Y\mathcal{M}, V \mapsto VY``, i.e. ``\mathfrak{g}^{\mathrm{ver},Y} = \{V\in\mathfrak{g}:VY = 0\}.``
" * Main.indentation * raw"2. The *horizontal component* ``\mathfrak{g}^{\mathrm{hor},Y}`` is the orthogonal complement of ``\mathfrak{g}^{\mathrm{ver},Y}`` in ``\mathfrak{g}``. It is isomorphic to ``T_Y\mathcal{M}``.
")
```

We will refer to the mapping from ``T_Y\mathcal{M}`` to ``\mathfrak{g}^{\mathrm{hor}, Y}`` by ``\Omega``. We will give explicit examples of ``\Omega`` below. If we have now defined a metric ``\langle\cdot,\cdot\rangle`` on ``\mathfrak{g}``, then this induces a Riemannian metric on ``\mathcal{M}``:
```math
g_Y(\Delta_1, \Delta_2) = \langle\Omega(Y,\Delta_1),\Omega(Y,\Delta_2)\rangle\text{ for $\Delta_1,\Delta_2\in{}T_Y\mathcal{M}$.}
```

Two examples of homogeneous spaces implemented in `GeometricMachineLearning` are the [Stiefel](@ref "The Stiefel Manifold") and the [Grassmann](@ref "The Grassmann Manifold") manifold. The Lie group ``SO(N)`` acts transitively on both of these manifolds, i.e. turns them into homogeneous spaces. The Lie algebra of ``SO(N)`` are the skew-symmetric matrices ``\mathfrak{so}(N):=\{V\in\mathbb{R}^{N\times{}N}:V^T + V = 0\}`` and the canonical metric associated with it is simply ``(V_1,V_2)\mapsto\frac{1}{2}\mathrm{Tr}(V_1^TV_2)``.


# The Stiefel Manifold 

The Stiefel manifold ``St(n, N)`` is the space of all orthonormal frames in ``\mathbb{R}^{N\times{}n}``, i.e. matrices ``Y\in\mathbb{R}^{N\times{}n}`` s.t. ``Y^TY = \mathbb{I}_n``. It can also be seen as ``SO(N)`` modulo an equivalence relation: ``A\sim{}B\iff{}AE = BE`` for 

```math
E = \begin{bmatrix}
\mathbb{I}_n \\ 
\mathbb{O}
\end{bmatrix}\in{}St(n, N),
```
which is the canonical element of the Stiefel manifold. In words: the first ``n`` columns of ``A`` and ``B`` are the same. We also use this principle to draw random elements from the Stiefel manifold.

```@eval
Main.example(raw"Drawing random elements from the Stiefel (and the Grassmann) manifold is done by first calling `rand(N, n)` (i.e. drawing from a normal distribution) and then performing a ``QR`` decomposition. We then take the first ``n`` columns of the ``Q`` matrix to be an element of the Stiefel manifold.")
```

The tangent space to the element ``Y\in{}St(n,N)`` can be determined by considering ``C^\infty`` curves on ``SO(N)`` through ``\mathbb{I}`` which we write ``t\mapsto{}A(t)``. Because ``SO(N)`` acts transitively on ``St(n, N)`` each ``C^\infty`` curve on ``St(n, N)`` through ``Y`` can be written as ``A(t)Y`` and we get: 

```math
T_YSt(n,N)=\{BY : B\in\mathfrak{g}\} = \{\Delta\in\mathbb{R}^{N\times{}n}: \Delta^TY + Y^T\Delta = \mathbb{O}\},
```

where the last equality can be established through the isomorphism 

```math
\Omega: T_YSt(n, N) \to \mathfrak{g}^{\mathrm{vec}, Y}, \Delta \mapsto (\mathbb{I} - \frac{1}{2}YY^T)\Delta{}Y^T - Y\Delta^T(\mathbb{I} - \frac{1}{2}YY^T).
```

That this is an isomorphism can be easily checked: 

```math
    \Omega(\Delta)Y = (\mathbb{I} - \frac{1}{2}YY^T)\Delta - \frac{1}{2}Y\Delta^TY = \Delta.
```

This isomorphism is also implemented in `GeometricMachineLearning`:

```@example
using GeometricMachineLearning

Y = rand(StiefelManifold{Float32}, 5, 3)
Δ = rgrad(Y, rand(Float32, 5, 3))
GeometricMachineLearning.Ω(Y, Δ) * Y.A ≈ Δ
```

The function `rgrad` is introduced below. 

## The Riemannian Gradient for the Stiefel Manifold

We defined the [Riemannian gradient](@ref "The Riemannian Gradient") to be a vector field ``\mathrm{grad}^gL`` such that it is *compatible with the Riemannian metric* in some sense; the definition we gave relied on an explicit coordinate chart. We can also express the Riemannian gradient for matrix manifolds by not relying on an explicit coordinate representation (which would be computationally expensive) [absil2004riemannian](@cite).

```@eval
Main.definition(raw"Given a Riemannian matrix manifold ``\mathcal{M}`` we define the **Riemannian gradient** of ``L:\mathcal{M}\to\mathbb{R}`` at ``Y``, called ``\mathrm{grad}_YL\in{}T_Y\mathcal{M}``, as the unique element of ``T_Y\mathcal{M}`` such that for any other ``\Delta\in{}T_Y\mathcal{M}`` we have
" * Main.indentation * raw"```math
" * Main.indentation * raw"\mathrm{Tr}((\nabla{}L)^T\Delta) = g_Y(\mathrm{grad}_YL, \Delta),
" * Main.indentation * raw"```
" * Main.indentation * raw"where Tr indicates the usual matrix trace.")
```

For the Stiefel manifold the Riemannian gradient is given by: 

```math
    \mathrm{grad}_YL = \nabla_YL - Y(\nabla_YL)^TY =: \mathtt{rgrad}(Y, \nabla_YL),
```

where ``\nabla_YL`` refers to the Euclidean gradient, i.e. 

```math
    [\nabla_YL]_{ij} = \frac{\partial{}L}{\partial{}y_{ij}}.
```

The Euclidean gradient ``\nabla{}L`` can in practice be obtained with an [AD routine](@ref "Pullbacks and Automatic Differentiation"). We then use the function `rgrad` to map ``\nabla_YL`` from ``\mathbb{R}^{N\times{}n}`` to ``T_YSt(n,N)``. We can check that this mapping indeed maps to the Riemannian gradient

```@example
using GeometricMachineLearning
using LinearAlgebra: tr

Y = rand(StiefelManifold{Float32}, 5, 3)
∇L = rand(Float32, 5, 3)
gradL = rgrad(Y, ∇L)
Δ = rgrad(Y, rand(Float32, 5, 3))

metric(Y, gradL, Δ) ≈ tr(∇L' * Δ)
```

# The Grassmann Manifold 

The Grassmann manifold is closely related to the Stiefel manifold, and an element of the Grassmann manifold can be represented through an element of the Stiefel manifold (but not vice-versa). An element of the Grassmann manifold ``G(n,N)`` is a vector subspace ``\subset\mathbb{R}^N`` of dimension $n$. Each such subspace (i.e. element of the Grassmann manifold) can be represented by a full-rank matrix ``A\in\mathbb{R}^{N\times{}n}`` and we identify two elements with the following equivalence relation: 

```math
    A_1 \sim A_2 \iff \exists{}C\in\mathbb{R}^{n\times{}n}\text{ s.t. }A_1C = A_2.
```

The resulting manifold is of dimension ``n(N-n)``. One can find a parametrization of the manifold the following way: Because the matrix ``Y`` has full rank, there have to be ``n`` independent columns in it: ``i_1, \ldots, i_n``. For simplicity assume that ``i_1 = 1, i_2=2, \ldots, i_n=n`` and call the matrix made up of these columns ``C``. Then the mapping to the coordinate chart is: ``YC^{-1}`` and the last ``N-n`` columns are the coordinates.

We can also define the Grassmann manifold based on the Stiefel manifold since elements of the Stiefel manifold are already full-rank matrices. In this case we have the following equivalence relation (for ``Y_1, Y_2\in{}St(n,N)``): 

```math
    Y_1 \sim Y_2 \iff \exists{}C\in{}SO(n)\text{ s.t. }Y_1C = Y_2.
```

In `GeometricMachineLearning` elements of the Grassmann manifold are drawn the same way as elements of the Stiefel manifold:

```@example
using GeometricMachineLearning

rand(GrassmannManifold{Float32}, 5, 3)
```

## The Riemannian Gradient of the Grassmann Manifold

Obtaining the Riemannian Gradient for the Grassmann manifold is slightly more difficult than it is in the case of the Stiefel manifold [absil2004riemannian](@cite). Since the Grassmann manifold can be obtained from the Stiefel manifold through an equivalence relation, we can however use this as a starting point. 

```@eval
Main.theorem(raw"The Riemannian gradient of a function ``L`` defined on the Grassmann manifold can be written as
" * Main.indentation * raw"```math
" * Main.indentation * raw"\mathrm{grad}_\mathcal{Y}^{Gr}L \simeq \nabla_Y{}L - YY^T\nabla_YL,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\nabla_Y{}L`` again is again the Euclidean gradient.")
```

```@eval
Main.proof(raw"In a first step we identify charts on the Grassmann manifold to make dealing with it easier. For this consider the following open cover of the Grassmann manifold. 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\{\mathcal{U}_W\}_{W\in{}St(n, N)} \quad\text{where}\quad \mathcal{U}_W = \{\mathrm{span}(Y):\mathrm{det}(W^TY)\neq0\}.
" * Main.indentation * raw"```
" * Main.indentation * raw"We can find a canonical bijective mapping from the set ``\mathcal{U}_W`` to the set ``\mathcal{S}_W := \{Y\in\mathbb{R}^{N\times{}n}:W^TY=\mathbb{I}_n\}``:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\sigma_W: \mathcal{U}_W \to \mathcal{S}_W,\, \mathcal{Y}=\mathrm{span}(Y)\mapsto{}Y(W^TY)^{-1} =: \hat{Y}.
" * Main.indentation * raw"```
" * Main.indentation * raw"That ``\sigma_W`` is well-defined is easy to see: Consider ``YC`` with ``C\in\mathbb{R}^{n\times{}n}`` non-singular. Then ``YC(W^TYC)^{-1}=Y(W^TY)^{-1} = \hat{Y}``. With this isomorphism we can also find a representation of elements of the tangent space:
" * Main.indentation * raw"```math
" * Main.indentation * raw"T_\mathcal{Y}\sigma_W: T_\mathcal{Y}Gr(n,N)\to{}T_{\hat{Y}}\mathcal{S}_W.
" * Main.indentation * raw"```
" * Main.indentation * raw"We give an explicit representation of this isomorphism; because the map ``\sigma_W`` does not care about the representation of ``\mathrm{span}(Y)`` we can perform the variations in ``St(n,N)``. We write the variations as ``Y(t)\in{}St(n,N)`` for ``t\in(-\varepsilon,\varepsilon)``. We also set ``Y(0) = Y`` and hence
" * Main.indentation * raw"```math
" * Main.indentation * raw"\frac{d}{dt}Y(t)(W^TY(t))^{-1} = (\dot{Y}(0) - Y(W^TY)^{-1}W^T\dot{Y}(0))(W^TY)^{-1},
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\dot{Y}(0)\in{}T_YSt(n,N)``. Also note  note that we have ``T_\mathcal{Y}\mathcal{U}_W = T_\mathcal{Y}Gr(n,N)`` because ``\mathcal{U}_W`` is an open subset of ``Gr(n,N)``. We thus can identify the tangent space ``T_\mathcal{Y}Gr(n,N)`` with the following set:
" * Main.indentation * raw"```math
" * Main.indentation * raw"T_{\hat{Y}}\mathcal{S}_W = \{(\Delta - YW^T\Delta)(W^T\Delta)^{-1}: Y\in{}St(n,N)\text{ s.t. }\mathrm{span}(Y)=\mathcal{Y}\text{ and }\Delta\in{}T_YSt(n,N)\}.
" * Main.indentation * raw"```
" * Main.indentation * raw"Further note that we can pick any element ``W`` to construct the charts for a neighborhood around the point ``\mathcal{Y}\in{}Gr(n,N)`` as long as we have ``\mathrm{det}(W^TY)\neq0`` for ``\mathrm{span}(Y)=\mathcal{Y}``. We  hence take ``W=Y`` and get the identification: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"T_\mathcal{Y}Gr(n,N) \equiv \{\Delta - YY^T\Delta: Y\in{}St(n,N)\text{ s.t. }\mathrm{span}(Y)=\mathcal{Y}\text{ and }\Delta\in{}T_YSt(n,N)\},
" * Main.indentation * raw"```
" * Main.indentation * raw"which is very easy to handle computationally (we simply store and change the matrix ``Y`` that represents an element of the Grassmann manifold). The Riemannian gradient is then 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\mathrm{grad}_\mathcal{Y}^{Gr}L = \mathrm{grad}_Y^{St}L - YY^T\mathrm{grad}_Y^{St}L = \nabla_Y{}L - YY^T\nabla_YL,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\mathrm{grad}^{St}_YL`` is the Riemannian gradient of the Stiefel manifold at ``Y``. We proved our assertion.")
```

 ## Library Functions 

```@docs
StiefelManifold
StiefelProjection
GrassmannManifold
rand(manifold_type::Type{MT}, ::Integer, ::Integer) where MT <: Manifold
GeometricMachineLearning.rgrad(::StiefelManifold, ::AbstractMatrix)
GeometricMachineLearning.rgrad(::GrassmannManifold, ::AbstractMatrix)
GeometricMachineLearning.metric(::StiefelManifold, ::AbstractMatrix, ::AbstractMatrix)
GeometricMachineLearning.metric(::GrassmannManifold, ::AbstractMatrix, ::AbstractMatrix)
GeometricMachineLearning.Ω(::StiefelManifold{T}, ::AbstractMatrix{T}) where T
GeometricMachineLearning.Ω(::GrassmannManifold{T}, ::AbstractMatrix{T}) where T
```

## References 

```@bibliography
Pages = []
Canonical = false

absil2004riemannian
frankel2011geometry
bendokat2021real
```