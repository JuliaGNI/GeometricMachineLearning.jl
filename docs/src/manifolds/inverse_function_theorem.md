# Foundational Theorem for Differential Manifolds

Here we state and proof all the theorem necessary to define [differential manifold](@ref "(Matrix) Manifolds"). All these theorems (including proofs) can be found in e.g. [lang2012fundamentals](@cite).

## The Fixed-Point Theorem 

The fixed-point theorem will be used in the proof of the inverse function theorem below and the [existence-and-uniqueness theorem](@ref "The Existence-And-Uniqueness Theorem"). 

```@eval
Main.theorem(raw"A function ``f:U \to U`` defined on an open subset ``U`` of a complete metric vector space ``\mathcal{V} \supset U`` that is contractive, i.e. ``|f(z) - f(y)| \leq q|z - y|`` with ``q < 1``, has a unique fixed point ``y^*`` such that ``f(y^*) = y^*``. Further ``y^*`` can be found by taking any ``y\in{}U`` through ``y^* = \lim_{m\to\infty}f^m(y)``."; name = "Banach Fixed-Point Theorem")
```

```@eval
Main.proof(raw"Fix a point ``y\in{}U``. We proof that the sequence ``(f^m(y))_{m\in\mathbb{N}}`` is Cauchy and because ``\mathcal{V}`` is a complete metric space, the limit of this sequence exists. Take ``\tilde{m} > m`` and we have
" *
Main.indentation * raw"```math
" *
Main.indentation * raw"\begin{aligned}
" *
Main.indentation * raw"|f^{\tilde{m}}(y) - f^m(y)| & \leq \sum_{i = m}^{\tilde{m} - 1}|f^{i+1}(y) - f^{i}(y)| \\
" *
Main.indentation * raw"                            & \leq \sum_{i = m}^{\tilde{m} - 1}q^i|f(y) - y| \\ 
" *
Main.indentation * raw"                            & \leq \sum_{i = m}^\infty{}q^i|f(y) - y|  = (f(y) - y)\left( \frac{q}{1 - q} - \sum_{i = 1}^{m-1}q^i \right)\\
" *
Main.indentation * raw"                            & = (f(y) - y)\left( \frac{q}{1 - q} - \frac{q - q^m}{q - 1} \right) = (f(y) - y)\frac{q^{m+1}}{1 - q}.
" *
Main.indentation * raw"\end{aligned} 
" *
Main.indentation * raw"```
" *
Main.indentation * raw"And the sequence is clearly Cauchy.")
```

Note that we stated the fixed-point theorem for arbitrary complete metric spaces here, not just for ``\mathbb{R}^n``. For the section on [manifolds](@ref "(Matrix) Manifolds") we only need the theorem for ``\mathbb{R}^n``, but for the [existence-and-uniqueness theorem](@ref "The Existence-And-Uniqueness Theorem") we need the statement for more general spaces. 


## The Inverse Function Theorem

The *inverse function theorem* gives a sufficient condition on a vector-valued function to be invertible in a neighborhood of a specific point. This theorem serves as a basis for the *implicit function theorem* and further for the [preimage theorem](@ref "The Preimage Theorem") and is critical in developing a theory of [manifolds](@ref "(Matrix) Manifolds"). Here we first state the theorem and then give a proof.

```@eval
Main.theorem(raw"Consider a vector-valued differentiable function ``F:\mathbb{R}^N\to\mathbb{R}^N`` and assume its Jacobian is non-degenerate at a point ``x\in\mathbb{R}^N``. Then there exists a neighborhood ``U`` that contains ``F(x)`` and on which ``F`` is invertible, i.e. ``\exists{}H:U\to\mathbb{R}^N`` s.t. ``\forall{}y\in{}U,\,F\circ{}H(y) = y`` and ``H`` is differentiable."; name = "Inverse function theorem")
```

```@eval
Main.proof(raw"""Consider a mapping ``F:\mathbb{R}^N\to\mathbb{R}^N`` and assume its Jacobian has full rank at point ``x``, i.e. ``\det{}F'(x)\neq0``. We further assume that ``F(x) = 0``, ``F'(x) = \mathbb{I}`` and ``x = 0``. Now consider a ball around ``x`` whose radius ``r`` we do not yet fix and two points ``y`` and ``z`` in that ball: ``y,z\in{}B(r)``. We further introduce the function ``G(y):=y-F(y)``. By the *mean value theorem* we have ``|G(y)| = |G(y) - x| = |G(y) - G(x)|\leq|y-x|\sup_{0<t<1}||G'(x + t(y-x))||`` where ``||\cdot||`` is the *operator norm*. Because ``t\mapsto{}G'(x+t(y-x))`` is continuous and ``G'(x)=0`` there must exist an ``r`` s.t. ``\forall{}t\in[0,1],\,||G'(x +t(y-x))||<1/2``. We have for any element ``y\in{}B(r)``: ``|G(y) | \leq ||G'(y)||\cdot|y| < |y|/2``, so ``G(B(r))\subset{}B(r/2)``. We further define ``G_z(y) := z + G(y)``; this map is contractive on ``B(r)`` (for ``z\in{}B(r/2)``): ``|G_z(y)| \leq |z| + |G(y) - x| < q < 1`` and therefore has a fixed point: ``y^* = G_z(y^*) = z + y^* - F(y^*)`` and we obtain ``z = F(y^*)``.  The inverse (which we call ``H:F(B(r/2))\to{}B(r)``) is also continuous by the last theorem presented in the [section on basic topological concepts](@ref "Basic Concepts from General Topology"). We now proof that the derivative of ``H`` at ``F(x) = 0`` exists and that it is equal to ``F'(H(z))^{-1}``. For this we let ``\eta\in{}F(B(r/2))`` go to zero. We further define ``\xi = F(z)`` and ``h = H(\xi + \eta) - z``:
""" * 
Main.indentation * raw"```math
" *
Main.indentation * raw"\begin{aligned}
" *
Main.indentation * raw"    |H(\xi+\eta) - H(\xi) - F'(z)^{-1}\eta| & = |h - F'(x)^{-1}\xi| = |h - F'(z)^{-1}(F(z + h) - \xi)| \\
" *
Main.indentation * raw"                                            & \leq ||F'(z)^{-1}||\cdot|F'(z)h - F(z + h) + \xi| \\
" *
Main.indentation * raw"                                            & \leq ||F'(z)^{-1}||\cdot|h|\cdot\left| F'(z)\frac{h}{|h|} - \frac{F(z + h) - \xi}{|h|} \right|,
" *
Main.indentation * raw"\end{aligned}
" *
Main.indentation * raw"```
" * 
Main.indentation * raw"and the rightmost expression is bounded because of the mean value theorem: ``F(z + h) - F(z) \leq sup_{0<t<1}|h| \cdot ||F'(z + th)||``.")
```

## The Implicit Function Theorem 

This theorem is a direct consequence of the inverse function theorem. 

```@eval
Main.theorem(raw"Given a function ``f:\mathbb{R}^{n+m}\to\mathbb{R}^n`` whose derivative at ``x\in\mathbb{R}^{n+m}`` has full rank, we can find a map ``h:U\to\mathbb{R}^{n+m}`` for a neighborhood ``U\ni(f(x), x_{n+1}, \ldots, x_{n+m})`` such that ``f\circ{}h`` is a projection onto the first factor, i.e. ``f(h(x_1, \ldots, x_{n+m})) = (x_1, \ldots, x_n).``"; name = "Implicit Function Theorem")
```

```@eval
Main.proof(raw"Consider the map ``x = (x_1, \ldots, x_{n+m}) = (f(x), x_{n+1}, \ldots, x_{n+m})``. The derivative of this map is clearly of full rank if ``f'(x)`` is of full rank and therefore invertible in a neighborhood around ``(f(x), x_{n+1}, \ldots, x_{n+m})``. We call this inverse map ``h``. We then see that ``f\circ{}h`` is a projection.")
```

The implicit function will be used to proof the [preimage theorem](@ref "The Preimage Theorem") which we use as a basis to construct all the manifolds in `GeometricMachineLearning`.

## References

```@bibliography
Pages = []
Canonical = false

lang2012fundamentals
```

