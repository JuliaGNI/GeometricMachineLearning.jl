# Structure-Preserving Neural Networks

What we means by a *structure-preserving neural network* or a *geometric neural network* is a modification of a standard neural networks such that it satisfies certain properties like [symplecticity](@ref "Symplectic Systems") or [volume preservation](@ref "Divergence-Free Vector Fields"). We first define standard neural networks:

```@eval
Main.definition(raw"A **neural network architecture** is a parameter-dependent realization of a function:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathrm{architecture}: \mathbb{P} \to \mathcal{C}(\mathcal{D}, \mathcal{M}), \Theta \mapsto \mathcal{NN}_\Theta,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\Theta`` are the *parameters of the neural network* (we call ``\mathbb{P}`` the parameter space). ``\mathbb{P}``, the domain space ``\mathcal{D}`` and the target space ``\mathcal{M}`` of the neural network may be spaces with arbitrary structure in general (i.e. need not be vector spaces).")
```

In this text the spaces ``\mathcal{D}`` and ``\mathcal{M}`` are vector spaces in most cases[^1]. The parameter space ``\mathbb{P}`` is however build [from manifolds in many cases](@ref "Neural Network Optimizers"). Weights have to be put on manifolds to realize [certain architectures that would otherwise not be possible](@ref "The Symplectic Autoencoder") and can make training [more efficient in other cases](@ref "MNIST Tutorial").

[^1]: One exception is [Grassmann learning](@ref "Example of a Neural Network with a Grassmann Layer") where we learn a vector space.

It is a classical result [hornik1989multilayer](@cite) that one-layer feedforward neural networks[^2] are *universal approximators*:

[^2]: We obtain one-layer feedforward neural networks by identifying ``\mathcal{P} = \mathbb{R}^{N\times{}n}\times\mathbb{R}^{N}\times\mathbb{R}^{m\times{}N}\ni(A, b, C) =: \Theta`` and ``\mathcal{NN}_\Theta(x) = C\sigma(Ax + b)`` for some scalar function ``\sigma:\mathbb{R}\to\mathbb{R}`` that is non-polynomial.

```@eval
Main.theorem(raw"Neural networks are dense in the space of continuous functions ``\mathcal{C}^(U, \mathbb{R}^m)`` in the *compact-open topology*, i.e. for every compact subset ``K\subset{}U,`` real number ``\varepsilon>0`` and function ``f\in\mathcal{C}(U, \mathbb{R}^m)`` we can find an integer ``N`` as well as weights
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \Theta = (A, b, C) \in \mathbb{R}^{N\times{}n}\times\mathbb{R}^{N}\times\mathbb{R}^{m\times{}N} =: \mathbb{P}
" * Main.indentation * raw"```
" * Main.indentation * raw"such that 
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \sup_{x \in K}|| f(x) - C\sigma(Ax + b) || < \varepsilon,
" * Main.indentation * raw"```
" * Main.indentation * raw"i.e. neural networks can approximate ``f`` arbitrarily well on any compact set ``K``.")
```

The universal approximation theorem has also been generalized to other neural network architectures [yun2019transformers, zhou2020universality, liu2024kan](@cite).

A *structure-preserving* or *geometric* neural networks is a neural network that has additional properties:

```@eval
Main.definition(raw"A **structure-preserving neural network architecture** is a parameter-dependent realization of a function:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathrm{sp}\cdot\mathrm{architecture}: \mathbb{P} \to \mathcal{C}(\mathcal{D}, \mathcal{M}), \Theta \mapsto \mathcal{NN}_\Theta,
" * Main.indentation * raw"```
" * Main.indentation * raw"such that ``\mathcal{NN}_\Theta`` preserves some structure.")
```

```@eval
Main.example(raw"We say that a neural network is **symplectic** if ``\mathcal{NN}_\Theta:\mathbb{R}^n\to\mathbb{R}^m`` (with ``m\geq{}n``) preserves ``\mathbb{J}``, i.e. 
" * Main.indentation * raw"```math
" * Main.indentation * raw"    (\nabla_z\mathcal{NN}_\Theta)^T\mathbb{J}_{2m}(\nabla_z\mathcal{NN}_\Theta) = \mathbb{J}_{2n},
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``z`` are coordinates on ``\mathbb{R}^n``.")
```

If we have ``m = n`` then we can use [SympNets](@ref "SympNet Architecture") to realize such architectures; SympNets furthermore are universal approximators for the set of canonical symplectic maps[^3] [jin2020sympnets](@cite). If ``m \neq n`` we can use [symplectic autoencoders](@ref "The Symplectic Autoencoder") to realize such an architecture. A different class of neural networks are *volume-preserving neural networks*:

[^3]: Other neural network architectures that were developed with the same aim are *Hamiltonian neural networks* [greydanus2019hamiltonian](@cite), *Hénon nets* [burby2020fast](@cite) and *generalized Hamiltonian neural networks* [horn2025generalized](@cite). [celledoni2021structure](@cite) gives an overview over *structure preserving neural networks*.

```@eval
Main.example(raw"We say that a neural network is **volume-preserving** if ``\mathcal{NN}_\Theta:\mathbb{R}^n\to\mathbb{R}^n`` is such that: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \det(\nabla_z\mathcal{NN}_\Theta) = 1,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``z`` are coordinates on ``\mathbb{R}^n``.")
```

Note that here we keep the dimension constant. Volume-preserving neural networks can be built on the basis of [feedforward neural networks](@ref "Volume-Preserving Feedforward Neural Network") or [transformers](@ref "Volume-Preserving Transformer").

```@raw latex
\section*{Chapter Summary}

In this chapter we introduced \textit{symplecticity} and \textit{volume presentation} (or divergence-freeness for the corresponding vector field) as examples of geometric structure. \textit{Symplecticity} is a property of the flow of Hamiltonian vector fields that dramatically restricts the accessible states of freedom. In many applications, neural networks, similar to classical numerical methods, aim at modeling the flow of a differential equation. Because symplecticity is a very restrictive property that we know the flow of the differential equation has, it is advantageous to also imbue the neural network with this property. We hence defined \textit{structure-preserving neural networks} as the ones that preserve symplecticity (or other structure) in this chapter.
\begin{comment}
```

## References

```@bibliography
Canonical = false
Pages = []

greydanus2019hamiltonian
burby2020fast
horn2025generalized
celledoni2021structure
```


```@raw latex
\end{comment}
```

```@raw html
<!--
```

# References

```@bibliography
Canonical = false
Pages = []

arnold1978mathematical
hairer2006geometric
leimkuhler2004simulating
greydanus2019hamiltonian
burby2020fast
horn2025generalized
celledoni2021structure
```


```@raw html
-->
```