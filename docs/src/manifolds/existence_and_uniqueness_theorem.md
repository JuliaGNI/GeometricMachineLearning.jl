# The Existence-And-Uniqueness Theorem

The *existence-and-uniqueness theorem*, also known as the *Picard-Lindelöf theorem*, *Picard's existence theorem* and the *Cauchy-Lipschitz theorem* gives a proof of the existence of solutions for ODEs. Here we state the existence-and-uniqueness theorem for manifolds as vector fields are just a special case of this. Its proof relies on the [Banach fixed-point theorem](@ref "The Fixed-Point Theorem").

```@eval
Main.theorem(raw"Let ``X`` a vector field on the manifold ``\mathcal{M}`` that is differentiable at ``x``. Then we can find an ``\epsilon>0`` and a unique curve ``\gamma:(-\epsilon, \epsilon)\to\mathcal{M}`` such that ``\gamma'(t) = X(\gamma(t))``."; name = "Existence-And-Uniqueness Theorem")
```

```@eval
Main.proof(raw"...")
``` 

## Reference

```@bibliography
Pages = []
Canonical = false

lang2012real
```