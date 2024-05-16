# Foundational Theorem for Differential Manifolds

Here we state and proof all the theorem necessary to define [differential manifold](@ref "(Matrix) Manifolds"). All these theorems (including proofs) can be found in e.g. [lang2012fundamentals](@cite).

## The Fixed Point Theorem 

The fixed point theorem will be used in the proof of the inverse function theorem below. 

```@eval
Main.theorem(raw"A function ``f:U \to U`` for ``U\subset\mathbb{R}^n`` that is contractive, i.e. ``|f(z) - f(y)| \leq q|z - y|`` with ``q < 1``, has a unique fixed point ``y^*`` such that ``f(y^*) = y^*``. Further ``y^*`` can be found by taking any ``y\in{}U`` through ``y^* = \lim_{m\to\infty}f^m(y)``."; name = "Banach Fixed Point Theorem")
```

__Proof__: Fix a point ``y\in{}U``. We proof that the sequence ``(f^m(y))_{m\in\mathbb{N}}`` is Cauchy and because ``\mathbb{R}^n`` is a complete metric space, the limit of this sequence exists. Take ``\tilde{m} > m`` and we have 
```math
|f^{\tilde{m}}(y) - f^m(y)| & \leq \sum_{i = m}^{\tilde{m} - 1}|f^{i+1}(y) - f^{i}(y)| \\
                            & \leq \sum_{i = m}^{\tilde{m} - 1}q^i|f(y) - y| \\ 
                            & \leq \sum_{i = m}^\infty{}q^i|f(y) - y|  = (f(y) - y)\left( \frac{q}{1 - q} - \sum_{i = 1}^{m-1}q^i \right)\\
                            & = (f(y) - y)\left( \frac{q}{1 - q} - \frac{q - q^m}{q - 1} \right) = (f(y) - y)\frac{q^{m+1}}{1 - q}.
```
And the sequence is clearly Cauchy. 


## The Inverse Function Theorem

The *inverse function theorem* gives a sufficient condition on a vector-valued function to be invertible in a neighborhood of a specific point. This theorem serves as a basis for the [submersion theorem](@ref "The Submersion Theorem") and is critical in developing a theory of [manifolds](@ref "(Matrix) Manifolds"). Here we first state the theorem and then give a proof.

```@eval
Main.theorem(raw"Consider a vector-valued differentiable function ``F:\mathbb{R}^N\to\mathbb{R}^N`` and assume its Jacobian is non-degenerate at a point ``x\in\mathbb{R}^N``. Then there exists a neighborhood ``U`` that contains ``F(x)`` and on which ``F`` is invertible, i.e. ``\exists{}H:U\to\mathbb{R}^N`` s.t. ``\forall{}y\in{}U,\,F\circ{}H(y) = y`` and ``H`` is differentiable."; name = "Inverse function theorem")
```

__Proof__: Consider a mapping ``F:\mathbb{R}^N\to\mathbb{R}^N`` and assume its Jacobian has full rank at point ``x``, i.e. ``\det{}F'(x)\neq0``. We further assume that ``F(x) = 0``, ``F'(x) = \mathbb{I}`` and ``x = 0``[^1]. Now consider a ball around ``x`` whose radius ``r`` we do not yet fix and two points ``y`` and ``z`` in that ball: ``y,z\in{}B(r)``. We further introduce the function ``G(y):=y-F(y)``. By the *mean value theorem* we have ``|G(y)| = |G(y) - x| = |G(y) - G(x)|\leq|y-x|\sup_{0<t<1}||G'(x + t(y-x))||`` where ``||\cdot||`` is the *operator norm*. Because ``t\mapsto{}G'(x+t(y-x))`` is continuous and ``G'(x)=0`` there must exist an ``r`` s.t. ``\forall{}t\in[0,1],\,||G'(x +t(y-x))||<1/2``. We have for any element ``y\in{}B(r)``: ``|G(y) | \leq ||G'(y)||\cdot|y| < |y|/2``, so ``G(B(r))\subset{}B(r/2)``. We further define ``G_z(y) := z + G(y)``; this map is contractive on ``B(r)`` (for ``z\in{}B(r/2)``): ``|G_z(y)| \leq |z| + |G(y) - x| < q < 1`` and therefore has a fixed point: ``y^* = G_z(y^*) = z + y^* - F(y^*)`` and we obtain ``z = F(y^*)``.  The inverse (which we call ``H:F(B(r/2))\to{}B(r)``) is also continuous by the last theorem presented in the [section on basic topological concepts](@ref "Basic Concepts from General Topology")[^2]. We now proof that the derivative of ``H`` at ``F(x) = 0`` exists and that it is equal to ``F'(H(z))^{-1}``. For this we let ``\eta\in{}F(B(r/2))`` go to zero. We further define ``\xi = F(z)`` and ``h = H(\xi + \eta) - z``:
```math
\begin{aligned}
    |H(\xi+\eta) - H(\xi) - F'(z)^{-1}\eta| & = |h - F'(x)^{-1}\xi| = |h - F'(z)^{-1}(F(z + h) - \xi)| \\
                                            & \leq ||F'(z)^{-1}||\cdot|F'(z)h - F(z + h) + \xi| \\
                                            & \leq ||F'(z)^{-1}||\cdot|h|\cdot\left| F'(z)\frac{h}{|h|} - \frac{F(z + h) - \xi}{|h|} \right|,
\end{aligned}
```
and the rightmost expression is bounded because of the mean value theorem: ``F(z + h) - F(z) \leq sup_{0<t<1}|h| \cdot ||F'(z + th)||``.

[^1]: We do not sacrifice generality in this as translation by a vector and multiplication by a full-rank matrix are invertible operations.

[^2]: In order to apply said theorem we must have a mapping from a compact space to a Hausdorff space. The image is clearly Hausdorff. For compactness, we could further restrict our ball to ``B(x,r/4)``, then ``H`` is at least continuous on the closure of ``F(B(x,r/4))``.

## The Implicit Function Theorem 

This theorem is a direct consequence of the inverse function theorem. 

```@eval
Main.theorem(""; name = "Implicit Function Theorem")
```

## The Submersion Theorem

The submersion theorem is an application of the inverse function theorem that we need in order to show that the spaces we deal with here are indeed manifolds. 

## References

```@bibliography
Pages = []
Canonical = false

lang2012fundamentals
```

