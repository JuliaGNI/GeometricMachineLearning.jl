# Divergence-Free Vector Fields

Hamiltonian vector fields are a subclass of so called divergence-free vector fields, i.e. the property of being divergence-free is less restrictive than the property of being Hamiltonian. We first define what it means to be divergence-free:

```@eval
Main.definition(raw"A vector field ``X:\mathcal{M}\to{}T\mathcal{M}`` defined on a Riemannian manifold ``(\mathcal{M}, g)`` is called **divergence-free** if ``\forall{}x\in\mathcal{M}`` we have:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\mathrm{div} (X)={\frac {1}{\rho }}{\frac {\partial \left({\frac {\rho }{\sqrt {g_{ii}}}}{\hat {X}}^{i}\right)}{\partial x^{i}}}={\frac {1}{\sqrt {\det g}}}{\frac {\partial \left({\sqrt {\frac {\det g}{g_{ii}}}}\,{\hat {X}}^{i}\right)}{\partial x^{i}}} = 0,
" * Main.indentation * raw"```
" * Main.indentation * raw"for some parametrization of a neighborhood around ``x``.")
```

We further define what it means to be volume-preserving:

```@eval
Main.definition(raw"We call a map ``\phi:\mathcal{M}\to\mathcal{M}`` **volume-preserving** if for all volume elements ``V\subset\mathcal{M}`` we have that ``\mathrm{vol}(V) = \mathrm{vol}(\phi(V)),`` where vol is a measure of a volume in a Riemannian manifold.")
```

We can proof the theorem: 

```@eval
Main.theorem(raw"The flow of a divergence-free vector field is volume-preserving.")
```

It is a classical result that *all Hamiltonian vector fields are divergence-free*, so volume-preservation is weaker than preservation of symplecticity [arnold1978mathematical](@cite).

The definition of integration on Riemannian manifolds and measures of volume elements can be found in standard textbooks on differential geometry, e.g. [bishop1980tensor, lang2012fundamentals](@cite).

## References

```@bibliography
Canonical = false
Pages = []

bishop1980tensor
lang2012fundamentals
arnold1978mathematical
```