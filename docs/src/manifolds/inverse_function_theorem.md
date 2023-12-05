# The Inverse Function Theorem

The **inverse function theorem** gives a sufficient condition on a vector-valued function to be invertible in a neighborhood of a specific point. This theorem is critical in developing a theory of [manifolds](manifolds.md) and serves as a basis for the [submersion theorem](submersion_theorem.md). Here we first state the theorem and then give a proof.

__Theorem (Inverse function theorem)__: Consider a vector-valued differentiable function ``F:\mathbb{R}^N\to\mathbb{R}^N`` and assume its Jacobian is non-degenerate at a point ``x\in\mathbb{R}^N``. Then there exists a neighborhood ``U`` that contains ``F(x)`` and on which ``F`` is invertible, i.e. ``\exists{}H:U\to\mathbb{R}^N`` s.t. ``\forall{}y\inU,\,F\circ{}H(y) = y`` and the inverse is differentiable.

__Proof__: Consider a mapping ``F:\mathbb{R}^N\to\mathbb{R}^N`` and assume its Jacobian has full rank at point ``x``, i.e. ``\det{}F'(x)\neq0``. Now consider a ball around ``x`` whose radius ``r`` we do not yet fix and two points ``y`` and ``z`` in that ball: ``y,z\in{}B(x,r)``. We further introduce the function ``G(y):=F(x)-F'(x)y``. By the *mean value theorem* we have ``|G(z) - G(y)|\leq|z-y|\sup_{0<t<1}||G'(x + t(y-x))||`` where ``||\cdot||`` is the *operator norm*. Because ``t\mapsto{}G'(x+t(y-x))`` is continuous and ``G'(x)=0`` there must exist an ``r`` s.t. ``\forall{}t\in[0,1],\,|G'(x +t(y-x)) - G'(x)|<\frac{1}{2}|F'(x)|``. ``F`` must then be injective on ``B(x,r)`` (and hence invertible on ``F(B(x,r))``). Assume for the moment it is not. We can then find two distinct elements ``y, z\in{}B(x,r)`` s.t. ``F(z) - F(y) = 0``. This implies ``|G(z) - G(y)| = ||F'(x)|||y - x|`` which is a contradiction.  The inverse (which we call ``H:F(B(x,r))\to{}B(x,r)``) is also continuous by the last theorem presented in the [section on basic topological concepts](basic_topology.md)[^1]. We still have to prove differentiability of the inverse. We now proof that the derivative of ``H`` at ``F(x)`` exists and that it is equal to ``F'(x)^{-1}F(x)``. For this we denote ``F(x)`` by ``\xi`` and let ``\eta\in{}F(B(x,r))`` go to zero.
```math
\begin{aligned}
    |\eta|^{-1}|H(\xi+\eta) - H(\xi) - F'(x)^{-1}\eta| & \leq |\eta|^{-1}||F'(x)||^{-1}|F'(x)H(\xi+\eta)-F'(x)H(\xi) -\eta| \\
                                            & \leq |\eta|^{-1}||F'(x)||^{-1}|F(H(\xi+\eta)) - G(H(\xi+\eta)) - F(H(\xi)) + G(x) - \eta| \\
                                            & = |\eta|^{-1}||F'(x)||^{-1}|\xi + \eta - G(H(\xi+\eta)) - \xi + G(x) - \eta| \\ 
                                            & = |\eta|^{-1}||F'(x)||^{-1}|G(H(\xi+\eta)) - G(H(\xi))|,
\end{aligned}
```
and this goes to zero as ``\eta`` goes to zero, because ``H`` is continuous and therefore ``H(\xi+\eta)`` goes to ``H(\xi)=x`` and the expression on the right goes to zero as well.

[^1]: In order to apply said theorem we must have a mapping from a compact space to a Hausdorff space. The image is clearly Hausdorff. For compactness, we could further restrict our ball to ``B(x,r/2)``, then ``G`` and its inverse are at least continuous on the closure of ``B(x,r/2)`` (or its image respectively) and hence also on ``B(x,r/2)``.


## References

<<<<<<< HEAD
```@bibliography
Pages = []
Canonical = false

lang2012fundamentals
```
=======
@book{lang2012fundamentals,
  title={Fundamentals of differential geometry},
  author={Lang, Serge},
  volume={191},
  year={2012},
  publisher={Springer Science \& Business Media}
}
>>>>>>> 9daa61fb1fc0177fc30f51b3df504196a03c688a
