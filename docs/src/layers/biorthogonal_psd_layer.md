# Biorthogonal PSD-Like Layers

A [`PSDLayer`](@ref) lifts ``\mathbb{R}^{2n}`` to ``\mathbb{R}^{2N}`` with one matrix ``\Phi\in{}St(n, N)``, applied to ``q`` and to ``p`` alike, and its inverse in the [`LeftInverseSymplecticAutoencoder`](@ref) is the transpose. A [`BiorthogonalPSDLayer`](@ref) applies two matrices, ``\Phi`` to ``q`` and ``\Psi`` to ``p``, that only have to be *biorthogonal*, ``\Psi^T\Phi = \mathbb{I}_n``. Biorthogonal weights of this kind are used in [otto2023learning, friedl2026learning](@cite).

The mathematics of the pairs is on the page [The Biorthogonal Manifold](@extref GeometricOptimizers The-Biorthogonal-Manifold) of `GeometricOptimizers`: there the lift ``A = \mathrm{diag}(\Phi, \Psi)`` is shown to be symplectic exactly when ``\Psi^T\Phi = \mathbb{I}_n`` ([A symplectic lift](@extref GeometricOptimizers A-symplectic-lift)), the pairs form the homogeneous space ``\mathcal{B}(n, N)\cong{}GL(N)/GL(N-n)`` of dimension ``2Nn - n^2`` ([A homogeneous space of the general linear group](@extref GeometricOptimizers A-homogeneous-space-of-the-general-linear-group)), and its global tangent space and retractions are given. This page says what the layers do with it.

## The layers

[`BiorthogonalPSDLayer`](@ref) applies ``A``, and [`TransposeBiorthogonalPSDLayer`](@ref) applies its symplectic inverse ``A^+ = \mathrm{diag}(\Psi^T, \Phi^T)``, i.e. ``\Psi^T`` to ``q`` and ``\Phi^T`` to ``p``; both docstrings below give the matrices. The [`PSDLayer`](@ref) is the case ``\Phi = \Psi\in{}St(n, N)``, where ``A^+ = A^T``.

`parameterlength` returns the dimension ``2Nn - n^2`` of ``\mathcal{B}(n, N)``, as `parameterlength` of a [`PSDLayer`](@ref) returns ``\dim{}St(n, N) = Nn - \frac{n(n+1)}{2}``. A biorthogonal lift therefore has ``Nn - \frac{n(n-1)}{2}`` more free parameters than the [`PSDLayer`](@ref) of the same size, slightly more than twice as many.

## The parameters

The layer has one parameter, `weight`, a point of [`BiorthogonalManifold`](@extref GeometricOptimizers GeometricOptimizers.BiorthogonalManifold), which stores the ``N\times{}2n`` matrix ``[\Phi\ \ \Psi]``. [`BiorthogonalPSDLayer`](@ref) multiplies by the two halves of that matrix and [`TransposeBiorthogonalPSDLayer`](@ref) by their transposes; neither inverts a matrix. ``\Psi^T\Phi = \mathbb{I}_n`` is a property of the point, and holds to the accuracy with which the point is on the manifold, which `check` of `GeometricOptimizers` measures as ``\|\Psi^T\Phi - \mathbb{I}_n\|``.

The layer is initialized at the pair ``(U, U)``, with ``U\in{}St(n, N)`` the weight that a [`PSDLayer`](@ref) of the same size draws from the same random number generator and initializer, so that it starts as that [`PSDLayer`](@ref).

The weight is trained on ``\mathcal{B}(n, N)`` by the manifold optimizers of `GeometricOptimizers`, as the weight of a [`PSDLayer`](@ref) is trained on the Stiefel manifold: the gradient that automatic differentiation returns for `weight` is the derivative with respect to the stored matrix ``[\Phi\ \ \Psi]``, the optimizer projects it onto the tangent space with `rgrad` and steps through the global tangent space of the manifold, and the retraction keeps ``\Psi^T\Phi = \mathbb{I}_n``. [`Adam`](@extref GeometricOptimizers GeometricOptimizers.Adam) takes entrywise second moments in the coordinates of the global section, and on this manifold its step depends on that section and need not descend; [`GradientMethod`](@extref GeometricOptimizers GeometricOptimizers.GradientMethod) and [`MomentumMethod`](@extref GeometricOptimizers GeometricOptimizers.MomentumMethod) do not depend on the section.

## In the left-inverse symplectic autoencoder

`LeftInverseSymplecticAutoencoder(full_dim, reduced_dim; biorthogonal = true)` uses a [`BiorthogonalPSDLayer`](@ref) for every lift of its decoder, and the encoder applies the [`TransposeBiorthogonalPSDLayer`](@ref)s, sharing their parameters. The encoder is still an exact left inverse of the decoder for every value of the parameters, since each layer of the encoder is the symplectic inverse of a layer of the decoder.

```@example
using GeometricMachineLearning

arch = LeftInverseSymplecticAutoencoder(10, 4; n_blocks = 2, n_layers = 2, biorthogonal = true)
nn = NeuralNetwork(arch)
ξ = rand(4)
encoder(nn)(decoder(nn)(ξ)) ≈ ξ
```

## Library Functions

```@docs
BiorthogonalPSDLayer
TransposeBiorthogonalPSDLayer
```
