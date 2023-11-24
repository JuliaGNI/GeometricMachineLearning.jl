# Basic Notions of Topology

On this page we discuss basic notions of topology that are necessary to define [manifolds](manifolds.md). Here we largely omit proofs and only define concepts that are necessary for defining a manifold[^1], namely the properties of being *Hausdorff* and *second countable*.

[^1]: Some authors (see e.g. [lang2012fundamentals](@cite)) do not require these properties. But since they constitute very weak restrictions and are always satisfied by the manifolds relevant for our purposes. Thus we require them here. 

__Definition__: A **topological space** is a set ``\mathcal{M}`` for which we define a collection of subsets of ``\mathcal{M}``, which we denote by ``\mathcal{T}`` and call the *open subsets*. ``\mathcal{T}`` further has to satisfy the following three conditions:
1. The empty set and ``\mathcal{M}`` belong to ``\mathcal{T}``.
2. Any union of an arbitrary number of elements of ``\mathcal{T}`` again belongs to ``\mathcal{T}``.
3. Any intersection of a finite number of elements of ``\mathcal{T}`` again belongs to ``\mathcal{T}``.

Based on this definition of a topological space we can now define what it means to be *Hausdorff*: 
__Definition__: A topological space ``\mathcal{M}`` is said to be **Hausdorff** if for any two points ``x,y\in\mathcal{M}`` we can find two open sets ``U_x,U_y\in\mathcal{T}`` s.t. ``x\in{}U_x, y\in{}U_y`` and ``U_x\cap{}U_y=\{\}``.

Lastly we define the notion of *second countability*:
__Definition__: A topological space ``\mathcal{M}`` is said to be **second-countable** if we can find a countable subcollection of ``\mathcal{T}`` called ``\mathcal{U}`` s.t. ``\forall{}U\in\mathcal{T}`` and ``x\in{}U`` we can find an element ``V\in\mathcal{U}`` for which ``x\in{}V\sub{}U``.


## References 
- General Topology book. 
- Lang

@book{lang2012fundamentals,
  title={Fundamentals of differential geometry},
  author={Lang, Serge},
  volume={191},
  year={2012},
  publisher={Springer Science \& Business Media}
}
