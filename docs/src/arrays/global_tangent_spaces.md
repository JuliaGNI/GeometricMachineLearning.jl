# Global Tangent Spaces

In `GeometricMachineLearning` standard neural network optimizers are generalized to [homogeneous spaces](@ref "Homogeneous Spaces") by leveraging the special structure of the tangent spaces of this class of manifolds. When we introduced homogeneous spaces we already talked about that every tangent space to a homogeneous space ``T_Y\mathcal{M}`` is of the form: 

```math
    T_Y\mathcal{M} = \mathfrak{g} \cdot Y := \{AY: A\in{}\mathfrak{g}\}.
```

We then have a decomposition of ``\mathfrak{g}`` into a vertical part ``\mathfrak{g}^{\mathrm{ver}, Y}`` and a horizontal part ``\mathfrak{g}^{\mathrm{hor}, Y}`` and the horizontal part is isomorphic to ``T_Y\mathcal{M}`` via:

```math
    \mathfrak{g}^{\mathrm{hor}, Y} = \{\Omega(\Delta): \Delta\in{}T_Y\mathcal{M} \}.
```

We now identify a special element ``E \in \mathcal{M}`` and designate the horizontal component ``\mathfrak{g}^{\mathrm{hor}, E}`` as our *global tangent space*. We will refer to this global tangent space by ``\mathfrak{g}^\mathrm{hor}``. We can now find a transformation from any ``\mathfrak{g}^{\mathrm{hor}, Y}`` to ``\mathfrak{g}^\mathrm{hor}`` and vice-versa (these spaces are isomorphic).

```@eval
Main.theorem(raw"Let ``A\in{}G`` an element such that ``AE = Y``. Then we have
" * Main.indentation * raw"```math
" * Main.indentation * raw"A^{-1}\cdot\mathfrak{g}^{\mathrm{hor},Y}\cdot{}A = \mathfrak{g}^\mathrm{hor},
" * Main.indentation * raw"```
" * Main.indentation * raw"i.e. for every element ``B\in\mathfrak{g}^\mathrm{hor}`` we can find a ``B^Y \in \mathfrak{g}^{\mathrm{hor},Y}`` s.t. ``B = A^{-1}B^YA`` (and vice-versa).")
```

```@eval
Main.proof(raw"We first show that for every ``B^Y\in\mathfrak{g}^{\mathrm{hor},Y}`` the element ``A^{-1}B^YA`` is in ``\mathfrak{g}^{\mathrm{hor}}``. First note that ``A^{-1}B^YA\in\mathfrak{g}`` by a fundamental theorem of Lie group theory (closedness of the Lie algebra under adjoint action). Now assume that ``A^{-1}B^YA`` is not fully contained in ``\mathfrak{g}^\mathrm{hor}``, i.e. it also has a vertical component. So we would lose information when performing ``A^{-1}B^YA \mapsto A^{-1}B^YAE = A^{-1}B^YY``, but this contradicts the fact that ``B^Y\in\mathfrak{g}^{\mathrm{hor},Y}.`` We now have to proof that for every ``B\in\mathfrak{g}^\mathrm{hor}`` we can find an element in ``\mathfrak{g}^{\mathrm{hor}, Y}`` such that this element is mapped to ``B``. By a argument similar to the one above we can show that ``ABA^{-1}\in\mathfrak{g}^\mathrm{hor, Y}`` and this element maps to ``B``. Proofing that the map is injective is now trivial.")
```

We should note that we have written all Lie group and Lie algebra actions as simple matrix multiplications, like ``AE = Y``. For some Lie groups and Lie algebras, as the Lie group of isomorphisms on some domain ``\mathcal{D}``, this notation may not be appropriate [holm2009geometric](@cite). These Lie groups are however not relevant for what we use in `GeometricMachineLearning` and we will stick to regular matrix notation.

## Global Sections 

Note that the theorem above requires us to find an element ``A\in{}G`` such that ``AE = Y``. We will call such a mapping ``\lambda:\mathcal{M}\to{}G`` a *global section*[^1]. 

[^1]: Global sections are also crucial for [parallel transport](@ref "Parallel Transport") in `GeometricMachineLearning`. A global section is first updated, i.e. ``\Lambda^{(t)} \gets \mathrm{update}(\Lambda^{(t-1)});`` and on the basis of this we then update the element of the manifold ``Y\in\mathcal{M}`` and the tangent vector ``\Delta\in{}T\mathcal{M}``.

```@eval
Main.definition(raw"We call a mapping ``\lambda`` from a homogeneous space ``\mathcal{M}`` to its associated Lie group ``G`` a **global section** if ``\forall{}Y\in\mathcal{M}`` it satisfies:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\lambda(Y)E = Y,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``E`` is the distinct element of the homogeneous space.")
```

Note that in general global sections are not unique because the rank of ``G`` is in general greater than that of ``\mathcal{M}``. We give an example of how to construct such a global section for the Stiefel and the Grassmann manifolds below. 

## The Global Tangent Space for the Stiefel Manifold

We now discuss the specific form of the global tangent space for the [Stiefel manifold](@ref "The Stiefel Manifold"). We pick as distinct element ``E`` (which build by calling [`StiefelProjection`](@ref)):

```math
E = \begin{bmatrix}
\mathbb{I}_n \\ 
\mathbb{O}
\end{bmatrix}\in{}St(n, N).
```

Based on this, elements of the vector space ``\mathfrak{g}^{\mathrm{hor}, E} =: \mathfrak{g}^{\mathrm{hor}}`` are: 

```math
\bar{B} = \begin{pmatrix}
A & B^T \\ B & \mathbb{O}
\end{pmatrix},
```

where ``A`` is a skew-symmetric matrix of size ``n\times{}n`` and ``B`` is an arbitrary matrix of size ``(N - n)\times{}n``. Arrays of type ``\mathfrak{g}^{\mathrm{hor}, E} \equiv \mathfrak{g}^\mathrm{hor}`` are implemented in `GeometricMachineLearning` under the name [`StiefelLieAlgHorMatrix`](@ref).

We can call this with e.g. a skew-symmetric matrix ``A`` and an arbitrary matrix ``B``:

```@example call_stiefel_lie_alg_hor_matrix_1
using GeometricMachineLearning # hide

N, n = 5, 2

A = rand(SkewSymMatrix, n)
```

```@example call_stiefel_lie_alg_hor_matrix_1
B = rand(N - n, n)
```

The constructor is then called as follows:

```@example call_stiefel_lie_alg_hor_matrix_1
B̄ = StiefelLieAlgHorMatrix(A, B, N, n)
```

We can also call it with a matrix of shape ``N\times{}N``:

```@example call_stiefel_lie_alg_hor_matrix_1
B̄₂ = Matrix(B̄) # note that this does not have any special structure

StiefelLieAlgHorMatrix(B̄₂, n)
```

Or we can call it on ``T_E\mathcal{M}\subset\mathbb{R}^{N\times{}n},`` i.e. a matrix of shape ``N\times{}n``:

```@example call_stiefel_lie_alg_hor_matrix_1
E = StiefelProjection(N, n)
```

```@example call_stiefel_lie_alg_hor_matrix_1
B̄₃ = B̄ * E

StiefelLieAlgHorMatrix(B̄₃, n)
```

We now demonstrate how to map from an element of ``\mathfrak{g}^{\mathrm{hor}, Y}`` to an element of ``\mathfrak{g}^\mathrm{hor}``:

```@example global_section
using GeometricMachineLearning # hide
using GeometricMachineLearning: Ω

N, n = 5, 2 # hide
Y = rand(StiefelManifold, N, n)
Δ = rgrad(Y, rand(N, n))
ΩΔ = Ω(Y, Δ)
λY = GlobalSection(Y) 

λY_mat = Matrix(λY)

round.(λY_mat' * ΩΔ * λY_mat; digits = 3)
```

Performing this computation directly is computationally very inefficient however and the user is strongly discouraged to call `Matrix` on an instance of [`GlobalSection`](@ref). The better option is calling [`global_rep`](@ref):

```@example global_section
using GeometricMachineLearning: _round # hide

_round(global_rep(λY, Δ); digits = 3)
```

Internally [`GlobalSection`](@ref) calls the function [`GeometricMachineLearning.global_section`](@ref) which does the following for the Stiefel manifold: 

```julia
A = randn(N, N - n) # or the gpu equivalent
A = A - Y * (Y' * A)
Y⟂ = qr(A).Q[1:N, 1:(N - n)]
```

So we draw ``(N - n)`` new columns randomly, subtract the part that is spanned by the columns of ``Y`` and then perform a ``QR`` composition on the resulting matrix. The ``Q`` part of the decomposition is a matrix of ``(N - n)`` columns that is orthogonal to ``Y`` and is typically referred to as ``Y_\perp``  [absil2004riemannian, absil2008optimization, bendokat2020grassmann](@cite). We can easily check that this ``Y_\perp`` is indeed orthogonal to ``Y``.

```@eval
Main.theorem(raw"The matrix ``Y_\perp`` constructed with the algorithm above satisfies
" * Main.indentation * raw"```math
" * Main.indentation * raw"Y^TY_\perp = \mathbb{O}_{n\times{}n},
" * Main.indentation * raw"```
" * Main.indentation * raw"and
" * Main.indentation * raw"```math
" * Main.indentation * raw"(Y_\perp)^TY_\perp = \mathbb{I}_n,
" * Main.indentation * raw"```
" * Main.indentation * raw"i.e. all the columns in the big matrix ``[Y, Y_\perp]\in\mathbb{R}^{N\times{}N}`` are mutually orthonormal and it therefore is an element of ``SO(N)``.")
```

```@eval
Main.proof(raw"The second property is trivially satisfied because the ``Q`` component of a ``QR`` decomposition is an orthogonal matrix. For the first property note that ``Y^TQR = \mathbb{O}`` is zero because we have subtracted the ``Y`` component from the matrix ``QR``. The matrix ``R\in\mathbb{R}^{N\times{}(N-n)}`` further has the property ``[R]_{ij} = 0`` for ``i > j`` and we have that 
" * Main.indentation * raw"```math
" * Main.indentation * raw"(Y^TQ)R = [r_{11}(Y^TQ)_{1\bullet}, r_{12}(Y^TQ)_{1\bullet} + r_{22}(Y^TQ)_{2\bullet}, \ldots, \sum_{i=1}^{N-n}r_{i(N-n)}(Y^TQ)_{i\bullet}].
" * Main.indentation * raw"```
" * Main.indentation * raw"Now all the coefficients ``r_{ii}`` are non-zero because the matrix we performed the ``QR`` decomposition on has full rank and we can see that if ``(Y^TQ)R`` is zero ``Y^TQ`` also has to be zero.")
```

The function [`global_rep`](@ref) furthermore makes use of the following:

```math
    \mathtt{global\_rep}(Y) = \lambda(Y)^T\Omega(Y,\Delta)\lambda(Y) = EY^T\Delta{}E^T + \begin{bmatrix} \mathbb{O} \\ \bar{\lambda}^T\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O} & E\Delta^T\bar{\lambda} \end{bmatrix},
```
where ``\lambda(Y) = [Y, \bar{\lambda}].``

```@eval
Main.proof(raw"We derive the expression above: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\begin{aligned}
" * Main.indentation * raw"\lambda(Y)^T\Omega(Y,\Delta)\lambda(Y)  & = \lambda(Y)^T[(\mathbb{I} - \frac{1}{2}YY^T)\Delta{}Y^T - Y\Delta^T(\mathbb{I} - \frac{1}{2}YY^T)]\lambda(Y) \\
" * Main.indentation * raw"                                        & = \lambda(Y)^T[(\mathbb{I} - \frac{1}{2}YY^T)\Delta{}E^T - Y\Delta^T(\lambda(Y) - \frac{1}{2}YE^T)] \\
" * Main.indentation * raw"                                        & = \lambda(Y)^T\Delta{}E^T - \frac{1}{2}EY^T\Delta{}E^T - E\Delta^T\lambda(Y) + \frac{1}{2}E\Delta^TYE^T \\ 
" * Main.indentation * raw"                                        & = \begin{bmatrix} Y^T\Delta{}E^T \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \frac{1}{2}EY^T\Delta{}E - \begin{bmatrix} E\Delta^TY & E\Delta^T\bar{\lambda} \end{bmatrix} + \frac{1}{2}E\Delta^TYE^T \\
" * Main.indentation * raw"                                        & = \begin{bmatrix} Y^T\Delta{}E^T \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} + E\Delta^TYE^T - \begin{bmatrix}E\Delta^TY & E\Delta^T\bar{\lambda} \end{bmatrix} \\
" * Main.indentation * raw"                                                & = EY^T\Delta{}E^T + E\Delta^TYE^T - E\Delta^TYE^T + \begin{bmatrix} \mathbb{O} \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O} & E\Delta^T\bar{\lambda} \end{bmatrix} \\
" * Main.indentation * raw"                                        & = EY^T\Delta{}E^T + \begin{bmatrix} \mathbb{O}_{n\times{}N} \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O}_{N\times{}n} & E\Delta^T\bar{\lambda} \end{bmatrix},
" * Main.indentation * raw"\end{aligned}
" * Main.indentation * raw"```
" * Main.indentation * raw"which proofs our assertion.")
```

This expression of [`global_rep`](@ref) means we only need ``Y^T\Delta`` and ``\bar{\lambda}^T\Delta`` and this is what is used internally.

We now discuss the global tangent space for the Grassmann manifold. This is similar to the Stiefel case.

## Global Tangent Space for the Grassmann Manifold

In the case of the Grassmann manifold we construct the global tangent space with respect to the distinct element ``\mathcal{E}=\mathrm{span}(E)\in{}Gr(n,N)``, where ``E`` is again the same matrix.

The tangent tangent space ``T_\mathcal{E}Gr(n,N)`` can be represented through matrices: 

```math
\begin{pmatrix}
    0 & \cdots & 0 \\
    \cdots & \cdots & \cdots \\ 
    0 & \cdots & 0 \\
    b_{11} & \cdots & b_{1n} \\
    \cdots & \cdots & \cdots \\ 
    b_{(N-n)1} & \cdots & b_{(N-n)n}
\end{pmatrix}.
```

This representation is based on the identification ``T_\mathcal{E}Gr(n,N)\to{}T_E\mathcal{S}_E`` that was discussed in the section on the [Grassmann manifold](@ref "The Grassmann Manifold")[^2]. We use the following notation:

[^2]: We derived the following expression for the Riemannian gradient of the Grassmann manifold: ``\mathrm{grad}_\mathcal{Y}^{Gr}L = \nabla_Y{}L - YY^T\nabla_YL``. The tangent space to the element ``\mathcal{E}`` can thus be written as ``\bar{B} - EE^T\bar{B}`` where ``B\in\mathbb{R}^{N\times{}n}`` and the matrices in this tangent space have the desired form. 

```math
\mathfrak{g}^\mathrm{hor} = \mathfrak{g}^{\mathrm{hor},\mathcal{E}} = \left\{\begin{pmatrix} 0 & -B^T \\ B & 0 \end{pmatrix}: \text{$B\in\mathbb{R}^{(N-n)\times{}n}$ is arbitrary}\right\}.
```

This is equivalent to the horizontal component of ``\mathfrak{g}`` for the Stiefel manifold for the case when ``A`` is zero. This is a reflection of the rotational invariance of the Grassmann manifold: the skew-symmetric matrices ``A`` are connected to the group of rotations ``O(n)`` which is factored out in the Grassmann manifold ``Gr(n,N)\simeq{}St(n,N)/O(n)``. In `GeometricMachineLearning` we thus treat the Grassmann manifold as being embedded in the Stiefel manifold. In [bendokat2020grassmann](@cite) viewing the Grassmann manifold as a quotient space of the Stiefel manifold is important for "feasibility" in "practical computations". 

## Library Functions

```@docs
GeometricMachineLearning.AbstractLieAlgHorMatrix
StiefelLieAlgHorMatrix
StiefelLieAlgHorMatrix(::AbstractMatrix, ::Int)
GrassmannLieAlgHorMatrix
GrassmannLieAlgHorMatrix(::AbstractMatrix, ::Int)
GlobalSection
Matrix(::GlobalSection)
apply_section
apply_section!
*(::GlobalSection, ::Manifold)
GeometricMachineLearning.global_section(::StiefelManifold{T}) where T
GeometricMachineLearning.global_section(::GrassmannManifold{T}) where T
global_rep
```


```@raw latex
\section*{Chapter Summary}

In this chapter we discussed mathematical core concepts in this dissertation: aspects from general topology and analysis. We furthermore discussed Riemannian geometry and parallel transport which are crucial concepts for manifold optimization. Homogeneous spaces were introduced as an important subcategory of manifolds and we showed how to find a global tangent space representation for them. This was done by identifying a distinct element $E\in\mathcal{M}$ and then considering the action of the Lie algebra $\mathfrak{g}$ on this element. We have presented our basic application user interface (API) for the \texttt{Julia} package \texttt{GeometricMachineLearning}. This API will be extended in Part II with a general optimizer framework. 
```

```@raw latex
\begin{comment}
```

## References

```@bibliography
Pages = []
Canonical = false

absil2004riemannian
absil2008optimization
bendokat2020grassmann
brantner2023generalizing
frankel2011geometry
```

```@raw latex
\end{comment}
```

```@raw html
<!--
```

# References

```@bibliography
Pages = []
Canonical = false 

lipschutz1965general
lang2012real
lang2012fundamentals
bishop1980tensor
do1992riemannian
absil2004riemannian
bendokat2020grassmann
frankel2011geometry
o1983semi
bendokat2021real
```

```@raw html
-->
```