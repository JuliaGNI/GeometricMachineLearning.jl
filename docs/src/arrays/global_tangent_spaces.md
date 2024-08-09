# Global Tangent Spaces

In `GeometricMachineLearning` standard neural network optimizers are generalized to [homogeneous spaces](@ref "Homogeneous Spaces") by leveraging the special structure of the tangent spaces of this class of manifolds. When we introduced homogeneous spaces we already talked about that every tangent space to a homogeneous space ``T_Y\mathcal{M}`` is of the form: 

```math
    T_Y\mathcal{M} = \mathfrak{g} \cdot Y := \{AY: A\in{}\mathfrak{g}\}.
```

We then have a decomposition of ``\mathfrak{g}`` into a vertical part ``\mathfrak{g}^{\mathrm{ver}, Y}`` and a horizontal part ``\mathfrak{g}^{\mathrm{hor}, Y}`` and the horizontal part is isomorphic to ``T_Y\mathcal{M}``. 

We now identify a special element ``E \in \mathcal{M}`` and designate the horizontal component ``\mathfrak{g}^{\mathrm{hor}, E}`` as our *global tangent space*. We will refer to this global tangent space by ``\mathfrak{g}^\mathrm{hor}``. We can now find a transformation from any ``\mathfrak{g}^{\mathrm{hor}, Y}`` to ``\mathfrak{g}^\mathrm{hor}`` and vice-versa (these spaces are isomorphic).

```@eval
Main.theorem(raw"Let ``A\in{}G`` an element such that ``AE = Y``. Then we have
" * Main.indentation * raw"```math
" * Main.indentation * raw"A^{-1}\cdot\mathfrak{g}^{\mathrm{hor},Y}\cdot{}A = \mathfrak{g}^\mathrm{hor},
" * Main.indentation * raw"```
" * Main.indentation * raw"i.e. for every element ``B\in\mathfrak{g}^\mathrm{hor}`` we can find a ``B^Y \in \mathfrak{g}^{\mathrm{hor},Y}`` s.t. ``B = A^{-1}B^YA`` (and vice-versa).")
```

```@eval
Main.proof(raw"We first show that for every ``B^Y\in\mathfrak{g}^{\mathrm{hor},Y}`` the element ``A^{-1}B^YA`` is in ``\mathfrak{g}^{\mathrm{hor}}``. First not that ``A^{-1}B^YA\in\mathfrak{g}`` by a fundamental theorem of Lie group theory (closedness of the Lie algebra under adjoint action). Now assume that ``A^{-1}B^YA`` is not fully contained in ``\mathfrak{g}^\mathrm{hor}``, i.e. it also has a vertical component. So we would lose information when performing ``A^{-1}B^YA \mapsto A^{-1}B^YAE = A^{-1}B^YY``, but this contradicts the fact that ``B^Y\in\mathfrak{g}^{\mathrm{hor},Y}.`` We now have to proof that for every ``B\in\mathfrak{g}^\mathrm{hor}`` we can find an element in ``\mathfrak{g}^{\mathrm{hor}, Y}`` such that this element is mapped to ``B``. By a argument similar to the one above we can show that ``ABA^{-1}\in\mathfrak{g}^\mathrm{hor, Y}`` and this element maps to ``B``. Proofing that the map is injective is now trivial.")
```

We should note that we have written all Lie group and Lie algebra actions as simple matrix multiplications, like ``AE = Y``. For some Lie groups and Lie algebras we should use different notations [holm2009geometric](@cite). These Lie groups are however not relevant for what we use in `GeometricMachineLearning` and we will stick to regular matrix notation.

## Global Sections 

Note that the theorem above requires us to find an element ``A\in{}G`` such that ``AE = Y``. If we can find a mapping ``\lambda:\mathcal{M}\to{}G`` we call such a mapping a *global section*. 

```@eval
Main.definition(raw"We call a mapping from ``\lambda:\mathcal{M} \to G`` a homogeneous space to its associated Lie group a **global section** if it satisfies:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\lambda(Y)E = Y,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``E`` is the distinct element of the homogeneous space.")
```

Note that in general global sections are not unique because the rank of ``G`` is in general greater than that of ``\mathcal{M}``. We give an example of how to construct such a global section for the Stiefel and the Grassmann manifolds below. 

Global sections are also crucial for [parallel transport](@ref "Parallel Transport") in `GeometricMachineLearning`. A global section is first updated:
```math
    \Lambda^{(t)} \gets \mathrm{update}(\Lambda^{(t-1)}),
```
and on the basis of this we then update the element of the manifold ``Y\in\mathcal{M}`` and the tangent vector ``\Delta\in{}T\mathcal{M}``.

## The Global Tangent Space for the Stiefel Manifold

We now discuss the specific form of the global tangent space for the [Stiefel manifold](@ref "The Stiefel Manifold"). We choose the distinct element[^1] ``E`` to have an especially simple form (this matrix can be build by calling [`StiefelProjection`](@ref)):

[^1]: We already introduced this special matrix together with the Stiefel manifold.

```math
E = \begin{bmatrix}
\mathbb{I}_n \\ 
\mathbb{O}
\end{bmatrix}\in{}St(n, N).
```

Based on this elements of the vector space ``\mathfrak{g}^{\mathrm{hor}, E} =: \mathfrak{g}^{\mathrm{hor}}`` are: 

```math
\begin{pmatrix}
A & B^T \\ B & \mathbb{O}
\end{pmatrix},
```

where ``A`` is a skew-symmetric matrix of size ``n\times{}n`` and ``B`` is an arbitrary matrix of size ``(N - n)\times{}n``.

Arrays of type ``\mathfrak{g}^{\mathrm{hor}, E}`` are implemented in `GeometricMachineLearning` under the name [`StiefelLieAlgHorMatrix`](@ref).

We can call this with e.g. a skew-symmetric matrix ``A`` and an arbitrary matrix ``B``:

```@example call_stiefel_lie_alg_hor_matrix_1
using GeometricMachineLearning # hide

N, n = 10, 4

A = rand(SkewSymMatrix, n)
```

```@example call_stiefel_lie_alg_hor_matrix_1
B = rand(N - n, n)
```

```@example call_stiefel_lie_alg_hor_matrix_1
B1 = StiefelLieAlgHorMatrix(A, B, N, n)
```

We can also call it with a matrix of shape ``N\times{}N``:

```@example call_stiefel_lie_alg_hor_matrix_1
B2 = Matrix(B1) # note that this does not have any special structure

StiefelLieAlgHorMatrix(B2, n)
```

Or we can call it a matrix of shape ``N\times{}n``:

```@example call_stiefel_lie_alg_hor_matrix_1
E = StiefelProjection(N, n)
```

```@example call_stiefel_lie_alg_hor_matrix_1
B3 = B1 * E

StiefelLieAlgHorMatrix(B3, n)
```

We now demonstrate how to map from an element of ``\mathfrak{g}^{\mathrm{hor}, Y}`` to an element of ``\mathfrak{g}^\mathrm{hor}``:

```@example global_section
using GeometricMachineLearning # hide

N, n = 10, 5

Y = rand(StiefelManifold, N, n)
Δ = rgrad(Y, rand(N, n))
ΩΔ = GeometricMachineLearning.Ω(Y, Δ)
λY = GlobalSection(Y) 

λY_mat = Matrix(λY)

round.(λY_mat' * ΩΔ * λY_mat; digits = 3)
```

Performing this computation directly is computationally very inefficient however and the user is strongly discouraged to call `Matrix` on an instance of [`GlobalSection`](@ref). The better option is calling [`global_rep`](@ref):

```@example global_section
using GeometricMachineLearning: _round # hide

_round(global_rep(λY, Δ); digits = 3)
```

Internally `GlobalSection` calls the function [`GeometricMachineLearning.global_section`](@ref) which does the following for the Stiefel manifold: 

```julia
A = randn(N, N - n) # or the gpu equivalent
A = A - Y * (Y' * A)
Y⟂ = qr(A).Q[1:N, 1:(N - n)]
```

So we draw ``(N - n)`` new columns randomly, subtract the part that is spanned by the columns of ``Y`` and then perform a ``QR`` composition on the resulting matrix. The ``Q`` part of the decomposition is a matrix of ``(N - n)`` columns that is orthogonal to ``Y`` and is typically referred to as ``Y_\perp``  [absil2004riemannian, absil2008optimization, bendokat2020grassmann](@cite). We can easily check that this ``Y_\perp`` is indeed orthogonal to ``Y``.

```@eval
Main.theorem(raw"The matrix ``Y_\perp`` constructed with the above algorithm satisfies
" * Main.indentation * raw"```math
" * Main.indentation * raw"Y^TY_\perp = \mathbb{O},
" * Main.indentation * raw"```
" * Main.indentation * raw"and
" * Main.indentation * raw"```math
" * Main.indentation * raw"(Y_\perp)^TY_\perp = \mathbb{I},
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
Main.proof(raw"In practice we use the following to make computations efficient: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\begin{aligned}
" * Main.indentation * raw"\lambda(Y)^T\Omega(Y,\Delta)\lambda(Y)  & = \lambda(Y)^T[(\mathbb{I} - \frac{1}{2}YY^T)\Delta{}Y^T - Y\Delta^T(\mathbb{I} - \frac{1}{2}YY^T)]\lambda(Y) \\
" * Main.indentation * raw"                                        & = \lambda(Y)^T[(\mathbb{I} - \frac{1}{2}YY^T)\Delta{}E^T - Y\Delta^T(\lambda(Y) - \frac{1}{2}YE^T)] \\
" * Main.indentation * raw"                                        & = \lambda(Y)^T\Delta{}E^T - \frac{1}{2}EY^T\Delta{}E^T - E\Delta^T\lambda(Y) + \frac{1}{2}E\Delta^TYE^T \\ 
" * Main.indentation * raw"                                        & = \begin{bmatrix} Y^T\Delta{}E^T \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \frac{1}{2}EY^T\Delta{}E - \begin{bmatrix} E\Delta^TY & E\Delta^T\bar{\lambda} \end{bmatrix} + \frac{1}{2}E\Delta^TYE^T \\
" * Main.indentation * raw"                                        & = \begin{bmatrix} Y^T\Delta{}E^T \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} + E\Delta^TYE^T - \begin{bmatrix}E\Delta^TY & E\Delta^T\bar{\lambda} \end{bmatrix} \\
" * Main.indentation * raw"                                                & = EY^T\Delta{}E^T + E\Delta^TYE^T - E\Delta^TYE^T + \begin{bmatrix} \mathbb{O} \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O} & E\Delta^T\bar{\lambda} \end{bmatrix} \\
" * Main.indentation * raw"                                        & = EY^T\Delta{}E^T + \begin{bmatrix} \mathbb{O} \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O} & E\Delta^T\bar{\lambda} \end{bmatrix},
" * Main.indentation * raw"\end{aligned},
" * Main.indentation * raw"```
" * Main.indentation * raw"which means we only need ``Y^T\Delta`` and ``\bar{\lambda}^T\Delta``.")
```

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

[^2]: We derived the following expression for the [Riemannian gradient of the Grassmann manifold](@ref "The Riemannian Gradient of the Grassmann Manifold"): ``\mathrm{grad}_\mathcal{Y}^{Gr}L = \nabla_Y{}L - YY^T\nabla_YL``. The tangent space to the element ``\mathcal{E}`` can thus be written as ``\bar{B} - EE^T\bar{B}`` where ``B\in\mathbb{R}^{N\times{}n}`` and the matrices in this tangent space have the desired form. 

```math
\mathfrak{g}^\mathrm{hor} = \mathfrak{g}^{\mathrm{hor},\mathcal{E}} = \left\{\begin{pmatrix} 0 & -B^T \\ B & 0 \end{pmatrix}: \text{$B$ arbitrary}\right\}.
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
GeometricMachineLearning.global_section
global_rep
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