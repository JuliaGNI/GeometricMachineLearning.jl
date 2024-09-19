```@raw latex
% This is a summary of the manifold chapter; this is only visible in the latex version
In this chapter we introduce basic concepts necessary to discuss manifolds and manifold optimization. We begin by discussing \textit{topological vector spaces} and \textit{topological metric spaces}, and several theorems important for developing a theory of manifolds such as the \textit{implicit function theorem}. We then define manifolds and discuss the \textit{preimage theorem} and the \textit{immersion theorem} as tools to give general spaces the structure of a manifold. We then proceed with a discussion on \textit{geodesics} and \textit{Riemannian manifolds}. The chapter concludes with a presentation of \textit{homogeneous spaces} and their \textit{global tangent space representation} that will be crucial for generalizing neural network optimizers to the manifold setting.
```

# Basic Concepts from General Topology

Here we discuss basic notions of topology that are necessary to define [manifolds](@ref "(Matrix) Manifolds") and work with them. Here we largely omit concrete examples and only define concepts that are necessary for defining a manifold[^1], namely the properties of being *Hausdorff* and *second countable*. For a detailed discussion of the theory and for a wide range of examples that illustrate this theory see e.g. [lipschutz1965general](@cite). The here-presented concepts are also (rudimentarily) covered in most differential geometry textbooks such as [lang2012fundamentals, bishop1980tensor](@cite). 


[^1]: Some authors (see e.g. [lang2012fundamentals](@cite)) do not require these properties. But since they constitute very weak restrictions and are always satisfied by the manifolds relevant for our purposes we require them here. 

We now start by giving all the definitions, theorem and corresponding proofs that are needed to define manifolds. Every manifold is a *topological space* which is why we give this definition first: 

```@eval
Main.definition(raw"A **topological space** is a set ``\mathcal{M}`` for which we are given a collection of subsets of ``\mathcal{M}``, which we denote by ``\mathcal{T}`` and call the *open subsets*. ``\mathcal{T}`` further has to satisfy the following three conditions:
" *
Main.indentation * raw"1. The empty set and ``\mathcal{M}`` belong to ``\mathcal{T}``.
" *
Main.indentation * raw"2. Any union of an arbitrary number of elements of ``\mathcal{T}`` again belongs to ``\mathcal{T}``.
" *
Main.indentation * raw"3. Any intersection of a finite number of elements of ``\mathcal{T}`` again belongs to ``\mathcal{T}``.
" *
Main.indentation * "So an arbitrary union of open sets is again open and a finite intersection of open sets is again open.")
```

Based on this definition of a topological space we can now define what it means to be *Hausdorff*: 

```@eval
Main.definition(raw"A topological space ``\mathcal{M}`` is said to be **Hausdorff** if for any two points ``x,y\in\mathcal{M}`` we can find two open sets ``U_x,U_y\in\mathcal{T}`` s.t. ``x\in{}U_x, y\in{}U_y`` and ``U_x\cap{}U_y=\{\}``.")
```

We now give the second definition that we need for defining manifolds, that of *second countability*:

```@eval
Main.definition(raw"A topological space ``\mathcal{M}`` is said to be **second-countable** if we can find a countable subcollection of ``\mathcal{T}`` called ``\mathcal{U}`` s.t. ``\forall{}U\in\mathcal{T}`` and ``x\in{}U`` we can find an element ``V\in\mathcal{U}`` for which ``x\in{}V\subset{}U``.")
```

We now give a few definitions and results that are needed for the [inverse function theorem](@ref "The Inverse Function Theorem") which is essential for practical applications of manifold theory. We start with the definition of *continuity*: 

```@eval
Main.definition(raw"A mapping ``f`` between topological spaces ``\mathcal{M}`` and ``\mathcal{N}`` is called **continuous** if the preimage of every open set is again an open set, i.e. if ``f^{-1}\{U\}\in\mathcal{T}`` for ``U`` open in ``\mathcal{N}`` and ``\mathcal{T}`` the topology on ``\mathcal{M}``.")
```

Continuity can also be formulated in terms of *closed sets* instead of doing it with *open sets*. The definition of closed sets is given below:

```@eval
Main.definition(raw"A **closed set** of a topological space ``\mathcal{M}`` is one whose complement is an open set, i.e. ``F`` is closed if ``F^c\in\mathcal{T}``, where the superscript ``{}^c`` indicates the complement: ``F^c := \{x\in\mathcal{M}:x\not\in{}F\}.`` For closed sets we thus have the following three properties:
" *
Main.indentation * raw"1. The empty set and ``\mathcal{M}`` are closed sets.
" *
Main.indentation * raw"2. Any union of a finite number of closed sets is again closed.
" *
Main.indentation * raw"3. Any intersection of an arbitrary number of closed sets is again closed.
" *
Main.indentation * "So a finite union of closed sets is again closed and an arbitrary intersection of closed sets is again closed.")
```

We now give the definition of continuity in terms of closed sets: 

```@eval
Main.theorem(raw"The definition of continuity in terms of open sets is equivalent to the following, second definition: ``f:\mathcal{M}\to\mathcal{N}`` is continuous if ``f^{-1}\{F\}\subset\mathcal{M}`` is a closed set for each closed set ``F\subset\mathcal{N}``.")
```

```@eval
Main.proof(raw"First assume that ``f`` is continuous according to the first definition and not to the second. Then ``f^{-1}\{F\}`` is not closed but ``f^{-1}\{F^c\}`` is open. But ``f^{-1}\{F^c\} = \{x\in\mathcal{M}:f(x)\not\in\mathcal{N}\} = (f^{-1}\{F\})^c`` cannot be open, else ``f^{-1}\{F\}`` would be closed. The implication of the first definition under assumption of the second can be shown analogously.")
```

The next theorem makes the rather abstract definition of *closed sets* more concrete; this definition is especially important for many practical proofs:

```@eval
Main.theorem(raw"The property of a set ``F`` being closed is equivalent to the following statement: If a point ``y`` is such that for every open set ``U`` containing it we have ``U\cap{}F\ne\{\}`` then this point is contained in ``F``.")
```

```@eval
Main.proof(raw"We first proof that if a set is closed then the statement holds. Consider a closed set ``F`` and a point ``y\not\in{}F`` s.t. every open set containing ``y`` has nonempty intersection with ``F``. But the complement ``F^c`` also is such a set, which is a clear contradiction. Now assume the above statement for a set ``F`` and further assume ``F`` is not closed. Its complement ``F^c`` is thus not open. Now consider the *interior* of this set: ``\mathrm{int}(F^c):=\cup\{U:U\subset{}F^c\text{ and $U$ open}\}``, i.e. the biggest open set contained within ``F^c``. Hence there must be a point ``y`` which is in ``F^c`` but is not in its interior, else ``F^c`` would be equal to its interior, i.e. would be open. We further must be able to find an open set ``U`` that contains ``y`` but is also contained in ``F^c``, else ``y`` would be an element of ``F``. A contradiction.")
```

Next we define *open covers*, a concept that is very important in developing a theory of manifolds: 

```@eval
Main.definition(raw"An **open cover** of a topological space ``\mathcal{M}`` is a (not necessarily countable) collection of open sets ``\{U_i\}_{i\mathcal{I}}`` s.t. their union contains ``\mathcal{M}``. A **finite open cover** is a finite collection of open sets that cover ``\mathcal{M}``. We say that an open cover is **reducible** to a finite cover if we can find a finite number of elements in the open cover whose union still contains ``\mathcal{M}``.")
```

And connected to this definition we state what it means for a topological space to be *compact*. This is a rather strong property that some of the manifolds treated in here have, for example the [Stiefel manifold](@ref "The Stiefel Manifold").

```@eval
Main.definition(raw"A topological space ``\mathcal{M}`` is called **compact** if every open cover is reducible to a finite cover.")
```

A very important result from general topology is that continuous functions preserve compactness[^2]: 

[^2]: We also say that *compactness is a topological property* [lipschutz1965general](@cite).

```@eval
Main.theorem(raw"Consider a continuous function ``f:\mathcal{M}\to\mathcal{N}`` and a compact set ``K\in\mathcal{M}``. Then ``f(K)`` is also compact.")
```

```@eval
Main.proof(raw"Consider an open cover of ``f(K)``: ``\{U_i\}_{i\in\mathcal{I}}``. Then ``\{f^{-1}\{U_i\}\}_{i\in\mathcal{I}}`` is an open cover of ``K`` and hence reducible to a finite cover ``\{f^{-1}\{U_i\}\}_{i\in\{i_1,\ldots,i_n\}}``. But then ``\{{U_i\}_{i\in\{i_1,\ldots,i_n}}`` also covers ``f(K)``.")
```

Moreover compactness is a property that is *inherited* by closed subspaces:

```@eval
Main.theorem(raw"A closed subset of a compact space is compact.")
```

```@eval
Main.proof(raw"Call the closed set ``F`` and consider an open cover of this set: ``\{U\}_{i\in\mathcal{I}}``. Then this open cover combined with ``F^c`` is an open cover for the entire compact space, hence reducible to a finite cover.")
```

If a set is contained in a Hausdorff space and is also compact we have:

```@eval
Main.theorem(raw"A compact subset of a Hausdorff space is closed.")
```

```@eval
Main.proof(raw"Consider a compact subset ``K``. If ``K`` is not closed, then there has to be a point ``y\not\in{}K`` s.t. every open set containing ``y`` intersects ``K``. Because the surrounding space is Hausdorff we can now find the following two collections of open sets: ``\{(U_z, U_{z,y}: U_z\cap{}U_{z,y}=\{\})\}_{z\in{}K}``. The open cover ``\{U_z\}_{z\in{}K}`` is then reducible to a finite cover ``\{U_z\}_{z\in\{z_1, \ldots, z_n\}}``. The intersection ``\cap_{z\in{z_1, \ldots, z_n}}U_{z,y}`` is then an open set that contains ``y`` but has no intersection with ``K``. A contraction.")
```

This last theorem we will use in proofing the [inverse function theorem](@ref "The Inverse Function Theorem"):

```@eval
Main.theorem(raw"If ``\mathcal{M}`` is compact and ``\mathcal{N}`` is Hausdorff, then the inverse of a continuous injective function ``f:\mathcal{M}\to\mathcal{N}`` is again continuous, i.e. ``f(V)`` is an open set in ``\mathcal{N}`` for ``V\in\mathcal{T}``.")
```

```@eval
Main.proof(raw"We can equivalently show that every closed set is mapped to a closed set. First consider the set ``K\in\mathcal{M}``. Its image is again compact and hence closed because ``\mathcal{N}`` is Hausdorff.")
```

We further define what it means for a set to be *dense*:

```@eval
Main.definition(raw"A set ``U`` is called **dense in ``D``**, where ``U\subset{}D`` if the *closure of ``U``*, i.e. the smallest closed set containing ``U``, also contains ``D``.")
```

We will come back to the notion of *denseness* when talking about the [universal approximation theorem for SympNets](@ref "Universal Approximation Theorems").

## References 

```@eval
references = raw"""
```@bibliography
Pages = []
Canonical = false 

lipschutz1965general
lang2012fundamentals
bishop1980tensor
```
"""
import Markdown
Main.output_type == :html ? Markdown.parse(references) : nothing
```
