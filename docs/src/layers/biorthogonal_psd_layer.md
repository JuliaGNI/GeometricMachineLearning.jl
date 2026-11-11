# Biorthogonal PSD-Like Layers

A [`PSDLayer`](@ref) lifts ``\mathbb{R}^{2n}`` to ``\mathbb{R}^{2N}`` with one matrix ``\Phi\in{}St(n, N)``, applied to ``q`` and to ``p`` alike, and its inverse in the [`LeftInverseSymplecticAutoencoder`](@ref) is the transpose. A [`BiorthogonalPSDLayer`](@ref) applies two matrices, ``\Phi`` to ``q`` and ``\Psi`` to ``p``, that only have to be *biorthogonal*, ``\Psi^T\Phi = \mathbb{I}_n``. Biorthogonal weights of this kind are used in [otto2023learning, friedl2026learning](@cite).

The mathematics of the pairs is on the page [The Biorthogonal Manifold](@extref GeometricOptimizers The-Biorthogonal-Manifold) of `GeometricOptimizers`: there the lift ``A = \mathrm{diag}(\Phi, \Psi)`` is shown to be symplectic exactly when ``\Psi^T\Phi = \mathbb{I}_n`` ([A symplectic lift](@extref GeometricOptimizers A-symplectic-lift)), the pairs form the homogeneous space ``\mathcal{B}(n, N)\cong{}GL(N)/GL(N-n)`` of dimension ``2Nn - n^2`` ([A homogeneous space of the general linear group](@extref GeometricOptimizers A-homogeneous-space-of-the-general-linear-group)), and the parametrization below is derived ([The bundle over the first factor](@extref GeometricOptimizers The-bundle-over-the-first-factor)). This page says what the layers do with it.

## The layers

[`BiorthogonalPSDLayer`](@ref) applies ``A``, and [`TransposeBiorthogonalPSDLayer`](@ref) applies its symplectic inverse ``A^+ = \mathrm{diag}(\Psi^T, \Phi^T)``, i.e. ``\Psi^T`` to ``q`` and ``\Phi^T`` to ``p``; both docstrings below give the matrices. The [`PSDLayer`](@ref) is the case ``\Phi = \Psi\in{}St(n, N)``, where ``A^+ = A^T``.

`parameterlength` returns the dimension ``2Nn - n^2`` of ``\mathcal{B}(n, N)``, as `parameterlength` of a [`PSDLayer`](@ref) returns ``\dim{}St(n, N) = Nn - \frac{n(n+1)}{2}``. A biorthogonal lift therefore has ``Nn - \frac{n(n-1)}{2}`` more free parameters than the [`PSDLayer`](@ref) of the same size, slightly more than twice as many.

## The parameters

The layer stores ``U`` (`weight`), ``R`` (`factor`) and ``W\in\mathbb{R}^{N\times{}n}`` (`complement`) and forms the pair as

```math
\Phi = UR, \qquad \Psi = UR^{-T} + (\mathbb{I}_N - UU^T)W, \qquad U\in{}St(n, N),\ R\in{}GL(n),
```

the parametrization of ``\mathcal{B}(n, N)`` of [The bundle over the first factor](@extref GeometricOptimizers The-bundle-over-the-first-factor), which reaches every pair and gives ``\Psi^T\Phi = \mathbb{I}_n`` for every ``U``, every invertible ``R`` and every ``W``. Only ``U`` is constrained. It is a [`StiefelManifold`](@extref GeometricOptimizers GeometricOptimizers.StiefelManifold) and is trained by the same manifold optimizers as the weight of a [`PSDLayer`](@ref); ``R`` and ``W`` are trained as Euclidean weights. The layer is initialized with ``R = \mathbb{I}_n`` and ``W = \mathbb{O}``, so that it starts as the [`PSDLayer`](@ref) with weight ``U``, and the layers apply ``R^{-1}`` without forming ``\Phi`` or ``\Psi``. In floating point ``\Psi^T\Phi = \mathbb{I}_n`` holds to round-off times the condition number of ``R``, and the layer requires ``R`` to stay invertible.

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
