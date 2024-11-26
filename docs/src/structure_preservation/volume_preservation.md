# Divergence-Free Vector Fields

The quality of being *divergence-free* greatly restricts the number of possible vector fields and also the dynamically accessible states of the flow map. It is however a weaker property then being [Hamiltonian](@ref "Symplectic Systems"). We first define what it means to be divergence-free:

```@eval
Main.definition(raw"A vector field ``X:\mathcal{M}\to{}T\mathcal{M}`` defined on a ``d``-dimensional Riemannian manifold ``(\mathcal{M}, g)`` is called **divergence-free** if ``\forall{}z\in\mathcal{M}`` we have:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\mathrm{div} (X)= \sum_{i = 1}^d {\frac {1}{\rho }}{\frac {\partial \left({\frac {\rho }{\sqrt {g_{ii}}}}{\hat {X}}^{i}\right)}{\partial z^{i}}}= \sum_{i = 1}^d {\frac {1}{\sqrt {\det g}}}{\frac {\partial \left({\sqrt {\frac {\det g}{g_{ii}}}}\,{\hat {X}}^{i}\right)}{\partial z^{i}}} = 0,
" * Main.indentation * raw"```
" * Main.indentation * raw"for some parametrization of a neighborhood around ``z``.")
```

```@eval
Main.remark(raw"If we do not deal with a general Riemannian manifold but simply with a vector space, the divergence for ``X:\mathbb{R}^d\to\mathbb{R}^d`` is usually written as
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathrm{div}(X) = \nabla\cdot{}X = \sum_{i=1}^d\frac{\partial{}X_i}{\partial{}z_i},
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``z`` are global coordinates.")
```

We further define what it means to be volume-preserving:

```@eval
Main.definition(raw"We call a map ``\phi:\mathcal{M}\to\mathcal{M}`` **volume-preserving** if for all volume elements ``V\subset\mathcal{M}`` we have that ``\mathrm{vol}(V) = \mathrm{vol}(\phi(V)),`` where vol is a measure of a volume in a Riemannian manifold.")
```

```@eval
Main.remark(raw"If we deal with vector spaces instead of more general manifolds the property of being **volume-preserving** in some domain ``\mathcal{D}\subset\mathbb{R}^d`` can be expressed as
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \det\nabla_z\varphi^t = 1 \quad \forall{}z\in\mathcal{D}, t\in[0, T].
" * Main.indentation * raw"```
" * Main.indentation * raw"So the columns of the Jacobian of ``\varphi^t`` span a volume element of size 1 for each ``z`` and each ``t``.")
```

We can proof the theorem: 

```@eval
Main.theorem(raw"The flow of a divergence-free vector field is volume-preserving.")
```

Here we only proof this statement for the case of a vector space. A proof of the more general statement can be found in standard textbooks on differential geometry[^1], e.g. [bishop1980tensor, lang2012fundamentals](@cite).

[^1]: Together with a precise definition of Riemannian integration and the volume form introduced above.

```@eval
Main.proof(raw"We refer to the flow of ``X`` by ``\varphi^t:\mathbb{R}^d\to\mathbb{R}^d`` and have the following property:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \frac{d}{dt}\nabla\varphi^t(z) = \nabla{}X(\varphi^t(z))\nabla\varphi^t(z).
" * Main.indentation * raw"```
" * Main.indentation * raw"Note that we used the convention ``[\nabla{}X(z)]_{ij} = \partial/\partial{}z_jX_i`` here. This expression for ``d/dt\nabla{}\varphi^t(x)`` further implies:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathrm{Tr}\left( (\nabla\varphi^t(z))^{-1}\frac{d}{dt}\nabla\varphi^t(z) \right) = \mathrm{Tr}\left( (\nabla\varphi^t(z))^{-1}\nabla{}X(\varphi^t(z))\nabla\varphi^t(z) \right) = \mathrm{Tr}(\nabla{}X(\varphi^t(z))) = 0,
" * Main.indentation * raw"```
" * Main.indentation * raw"where we have used ``\mathrm{Tr}(ABC) = \mathrm{Tr}(BCA)`` in the second equality, and ``\mathrm{Tr}(\nabla{}X) = \sum_{i=1}^d\partial{}X_i/\partial{}z_i = \mathrm{div}(X)`` and the divergence-freeness of ``X`` in the third equality. We further have
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathrm{Tr}(A^{-1}\dot{A}) = \frac{\frac{d}{dt}\mathrm{det}(A)}{\mathrm{det}(A)},
" * Main.indentation * raw"```
" * Main.indentation * raw"which can be derived from the classical result ``\mathrm{det}(\mathrm{exp}(A)) = \mathrm{exp}(\mathrm{Tr}(A)).`` Hence we have
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \frac{d}{dt}\mathrm{det}(\nabla\varphi^t(z)) = 0,
" * Main.indentation * raw"```
" * Main.indentation * raw"and the result is proved.")
```


It is a classical result that *all Hamiltonian vector fields are divergence-free*, so volume-preservation is weaker than preservation of symplecticity [arnold1978mathematical](@cite).

```@raw latex
\begin{comment}
```

## References

```@bibliography
Canonical = false
Pages = []

bishop1980tensor
lang2012fundamentals
arnold1978mathematical
```

```@raw latex
\end{comment}
```