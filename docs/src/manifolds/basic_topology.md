# Basic Concepts of General Topology

On this page we discuss basic notions of topology that are necessary to define and work [manifolds](manifolds.md). Here we largely omit concrete examples and only define concepts that are necessary for defining a manifold[^1], namely the properties of being *Hausdorff* and *second countable*. For a wide range of examples and a detailed discussion of the theory see e.g. [lipschutz1965general](@cite). The here-presented theory is also (rudimentary) covered in most differential geometry books such as [lang2012fundamentals](@cite) and [bishop1980tensor](@cite). 

[^1]: Some authors (see e.g. [lang2012fundamentals](@cite)) do not require these properties. But since they constitute very weak restrictions and are always satisfied by the manifolds relevant for our purposes we require them here. 

__Definition__: A **topological space** is a set ``\mathcal{M}`` for which we define a collection of subsets of ``\mathcal{M}``, which we denote by ``\mathcal{T}`` and call the *open subsets*. ``\mathcal{T}`` further has to satisfy the following three conditions:
1. The empty set and ``\mathcal{M}`` belong to ``\mathcal{T}``.
2. Any union of an arbitrary number of elements of ``\mathcal{T}`` again belongs to ``\mathcal{T}``.
3. Any intersection of a finite number of elements of ``\mathcal{T}`` again belongs to ``\mathcal{T}``.

Based on this definition of a topological space we can now define what it means to be *Hausdorff*: 
__Definition__: A topological space ``\mathcal{M}`` is said to be **Hausdorff** if for any two points ``x,y\in\mathcal{M}`` we can find two open sets ``U_x,U_y\in\mathcal{T}`` s.t. ``x\in{}U_x, y\in{}U_y`` and ``U_x\cap{}U_y=\{\}``.

We now give the second definition that we need for defining manifolds, that of *second countability*:
__Definition__: A topological space ``\mathcal{M}`` is said to be **second-countable** if we can find a countable subcollection of ``\mathcal{T}`` called ``\mathcal{U}`` s.t. ``\forall{}U\in\mathcal{T}`` and ``x\in{}U`` we can find an element ``V\in\mathcal{U}`` for which ``x\in{}V\sub{}U``.

We now give a few definitions and results that are needed for the [inverse function theorem](inverse_function_theorem.md) which is essential for practical applications of manifold theory.

__Definition__: A mapping ``f`` between topological spaces ``\mathcal{M}`` and ``\mathcal{N}`` is called **continuous** if the preimage of every open set is again an open set, i.e. if ``f^{-1}\{U\}\in\mathcal{T}`` for ``U`` open in ``\mathcal{N}`` and ``\mathcal{T}`` the topology on ``\mathcal{M}``.

__Definition__: A **closed set** of a topological space ``\mathcal{M}`` is one whose complement is an open set, i.e. ``F`` is closed if ``F^c\in\mathcal{T}``, where the superscript ``{}^c`` indicates the complement. For closed sets we thus have the following three properties: 
1. The empty set and ``\mathcal{M}`` are closed sets.
2. Any union of a finite number of closed sets is again closed.
3. Any intersection of an arbitrary number of closed sets is again closed.

__Theorem__: The definition of continuity is equivalent to the following, second definition: ``f:\mathcal{M}\to\mathcal{N}`` is continuous if ``f^{-1}\{F\}\sub\mathcal{M}`` is a closed set for each closed set ``F\sub\mathcal{N}``.

__Proof__: First assume that ``f`` is continuous according to the first definition and not to the second. Then ``f^{-1}{F}`` is not closed but ``f^{-1}{F^c}`` is open. But ``f^{-1}\{F^c\} = \{x\in\mathcal{M}:f(x)\nin\mathcal{N}\} = (f^{-1}\{F\})^c`` cannot be open, else ``f^{-1}\{F\}`` would be closed. The implication of the first definition under assumption of the second can be shown analogously. 

__Theorem__: The property of a set ``F`` being closed is equivalent to the following statement: If a point ``y`` is such that for every open set ``U`` containing it we have ``U\cap{}F\neq\{\}`` then this point is contained in ``F``.

__Proof__: We first proof that if a set is closed then the statement holds. Consider a closed set ``F`` and a point ``y\nin{}F`` s.t. every open set containing ``y`` has nonempty intersection with ``F``. But the complement ``F^c`` also is such a set, which is a clear contradiction. Now assume the above statement for a set ``F`` and further assume ``F`` is not closed. Its complement ``F^c`` is thus not open. Now consider the *interior* of this set: ``\mathrm{int}(F^c):=\cup\{U:U\sub{}F^c\}``, i.e. the biggest open set contained within ``F^c``. Hence there must be a point ``y`` which is in ``F^c`` but is not in its interior, else ``F^c`` would be equal to its interior, i.e. would be open. We further must be able to find an open set ``U`` that contains ``y`` but is also contained in ``F^c``, else ``y`` would be an element of ``F``. A contradiction. 

__Definition__: An **open cover** of a topological space ``\mathcal{M}`` is a (not necessarily countable) collection of open sets ``\{U_i\}_{i\mathcal{I}}`` s.t. their union contains ``\mathcal{M}``. A **finite open cover** is a collection of a finite number of open sets that cover ``\mathcal{M}``. We say that an open cover is **reducible** to a finite cover if we can find a finite number of elements in the open cover whose union still contains ``\mathcal{M}``.

__Definition__: A topological space ``\mathcal{M}`` is called **compact** if every open cover is reducible to a finite cover.

__Theorem__: Consider a continuous function ``f:\mathcal{M}\to\mathcal{N}`` and a compact set ``K\in\mathcal{M}``. Then ``f(K)`` is also compact. 

__Proof__: Consider an open cover of ``f(K)``: ``\{U_i\}_{i\in\mathcal{I}}``. Then ``\{f^{-1}\{U_i\}\}_{i\in\mathcal{I}}`` is an open cover of ``K`` and hence reducible to a finite cover ``\{f^{-1}\{U_i\}\}_{i\in\{i_1,\ldots,i_n}}``. But then ``\{{U_i\}_{i\in\{i_1,\ldots,i_n}}`` also covers ``f(K)``.

__Theorem__: A closed subset of a compact space is compact:

__Proof__: Call the closed set ``F`` and consider an open cover of this set: ``\{U\}_{i\in\mathcal{I}}``. Then this open cover combined with ``F^c`` is an open cover for the entire compact space, hence reducible to a finite cover.

__Theorem__: A compact subset of a Hausdorff space is closed: 

__Proof__: Consider a compact subset ``K``. If ``K`` is not closed, then there has to be a point ``y\nin{}K`` s.t. every open set containing ``y`` intersects ``K``. Because the surrounding space is Hausdorff we can now find the following two collections of open sets: ``\{(U_z, U_{z,y}: U_z\cap{}U_{z,y}=\{\})\}_{z\in{}K}``. The open cover ``\{U_z}_{z\in{}K}`` is then reducible to a finite cover ``\{U_z}_{z\in{z_1, \ldots, z_n}\}``. The intersection ``\cap_{z\in{z_1, \ldots, z_n}}U_{z,y}`` is then an open set that contains ``y`` but has no intersection with ``K``. A contraction. 

__Theorem__: If ``\mathcal{M}`` is compact and ``\mathcal{N}`` is Hausdorff, then the inverse of a continuous function ``f:\mathcal{M}\to\mathcal{N}`` is again continuous, i.e. ``f(V)`` is an open set in ``\mathcal{N}`` for ``V\in\mathcal{T}``.

__Proof__: We can equivalently show that every closed set is mapped to a closed set. First consider the set ``K\in\mathcal{M}``. Its image is again compact and hence closed because ``\mathcal{N}`` is Hausdorff. 

## References 

```@bibliography
Pages = []
Canonical = false 

bishop1980tensor
lang2012fundamentals
lipschutz1965general
```