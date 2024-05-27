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

We should note that we have written all Lie group and Lie algebra actions as simple matrix multiplications, like ``AE = Y``. For some Lie groups and Lie algebras we should use different notations [holm2009geometric](@cite). But this is not relevant for what we use in `GeometricMachineLearning`.

## Global Sections 

Note that the theorem above requires us to find an element ``A\in{}G`` such that ``AE = Y``. If we can find a mapping ``\lambda:\mathcal{M}\to{}G`` we call such a mapping a *global section*. Note that in general global sections are not unique because the rank of ``G`` is in general greater than that of ``\mathcal{M}``. We give an example of how to construct such a global section for the Stiefel and the Grassmann manifold below. 

## The Global Tangent Space for the Stiefel Manifold

We now discuss the specific form of the global tangent space for the [Stiefel manifold](@ref "The Stiefel Manifold"). We choose the distinct element[^1] ``E`` to have an especially simple form:

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

Arrays of type ``\mathfrak{g}^{\mathrm{hor}, E}`` are implemented in `GeometricMachineLearning` under the name `StiefelLieAlgHorMatrix`.

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

For computational reasons the user is strongly discouraged to call `Matrix` on an instance of `GlobalSection`. The better option is:

```@example global_section
using GeometricMachineLearning: _round # hide

_round(global_rep(λY, Δ); digits = 3)
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

This is equivalent to the horizontal component of ``\mathfrak{g}`` for the Stiefel manifold for the case when ``A`` is zero. This is a reflection of the rotational invariance of the Grassmann manifold: the skew-symmetric matrices ``A`` are connected to the group of rotations ``O(n)`` which is factored out in the Grassmann manifold ``Gr(n,N)\simeq{}St(n,N)/O(n)``.

## Library Functions

```@docs; canonical=false
StiefelLieAlgHorMatrix
GrassmannLieAlgHorMatrix
global_rep
```


## References

```@bibliography
Pages = []
Canonical = false

brantner2023generalizing
bendokat2020grassmann
```