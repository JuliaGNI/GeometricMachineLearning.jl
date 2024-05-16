# (Matrix) Manifolds

Manifolds are topological spaces that locally look like vector spaces. In the following we restrict ourselves to finite-dimensional smooth[^1] manifolds. 

[^1]: *Smooth* here means ``C^\infty``.

```@eval 
Main.theorem(raw"A **finite-dimensional smooth manifold** of dimension ``n`` is a second-countable Hausdorff space ``\mathcal{M}`` for which ``\forall{}x\in\mathcal{M}`` we can find a neighborhood ``U`` that contains ``x`` and a corresponding homeomorphism ``\varphi_U:U\cong{}W\subset\mathbb{R}^n`` where ``W`` is an open subset. The homeomorphisms ``\varphi_U`` are referred to as *coordinate charts*. If two such coordinate charts overlap, i.e. if ``U_1\cap{}U_2\neq\{\}``, then the map ``\varphi_{U_2}^{-1}\circ\varphi_{U_1}`` is ``C^\infty``.")
```

One example of a manifold that is also important for `GeometricMachineLearning` is the Lie group[^2] of orthonormal matrices ``SO(N)``. Before we can proof that ``SO(N)`` is a manifold we first need  another definition and a theorem:

[^2]: Lie groups are manifolds that also have a *group structure*, i.e. there is an operation ``\mathcal{M}\times\mathcal{M}\to\mathcal{M},(a,b)\mapsto{}ab`` s.t. ``(ab)c = a(bc)`` and  there exists a neutral element``e\mathcal{M}`` s.t. ``ae`` = ``a`` ``\forall{}a\in\mathcal{M}`` as well as an (for every ``a``) inverse element ``a^{-1}`` s.t. ``a(a^{-1}) = e``. The neutral element ``e`` we refer to as ``\mathbb{I}`` when dealing with matrix manifolds.

```@eval
Main.definition(raw"Consider a smooth mapping ``g: \mathcal{M}\to\mathcal{N}`` from one manifold to another. A point ``B\in\mathcal{M}`` is called a **regular value** of ``\mathcal{M}`` if ``\forall{}A\in{}g^{-1}\{B\}`` the map ``T_Ag:T_A\mathcal{M}\to{}T_{g(A)}\mathcal{N}`` is surjective.")
```

```@eval
Main.theorem(raw"Consider a smooth map ``g:\mathcal{M}\to\mathcal{N}`` from one manifold to another. Then the preimage of a regular point ``B`` of ``\mathcal{N}`` is a submanifold of ``\mathcal{M}``. Furthermore the codimension of ``g^{-1}\{B\}`` is equal to the dimension of ``\mathcal{N}`` and the tangent space ``T_A(g^{-1}\{B\})`` is equal to the kernel of ``T_Ag``."; name = "Preimage Theorem")
```

__Proof__: 


```@eval
Main.theorem(raw"The group ``SO(N)`` is a Lie group (i.e. has manifold structure).")
```

__Proof__: The vector space ``\mathbb{R}^{N\times{}N}`` clearly has manifold structure. The group ``SO(N)`` is equivalent to one of the level sets of the mapping: ``f:\mathbb{R}^{N\times{}N}\to\mathcal{S}(N), A\mapsto{}A^TA``, i.e. it is the component of ``f^{-1}\{\mathbb{I}\}`` that contains ``\mathbb{I}``. We still need to proof that ``\mathbb{I}`` is a regular point of ``f``, i.e. that for ``A\in{}SO(N)`` the mapping ``T_Af`` is surjective. This means that ``\forall{}B\in\mathcal{S}(N), A\in\mathbb{R}^{N\times{}N}`` ``\exists{}C\in\mathbb{R}^{N\times{}N}`` s.t. ``C^TA + A^TC = B``. The element ``C=\frac{1}{2}AB\in\mathcal{R}^{N\times{}N}`` satisfies this property.

With the definition above we can generalize the notion of an ordinary differential equation (ODE) on a vector space to an ordinary differential equation on a manifold:

__Definition__: An **ODE on a manifold** is a mapping that assigns to each element of the manifold ``A\in\mathcal{M}`` an element of the corresponding tangent space ``T_A\mathcal{M}``.

## Library Functions 

```@docs; canonical=false
Manifold
```

## References 

```@bibliography
Pages = []
Canonical = false

absil2008optimization
```