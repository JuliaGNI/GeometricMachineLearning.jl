# Biorthogonal PSD-Like Layers

A [`PSDLayer`](@ref) lifts ``\mathbb{R}^{2n}`` to ``\mathbb{R}^{2N}`` with one matrix ``\Phi\in{}St(n, N)``, applied to ``q`` and to ``p`` alike, and its inverse in the [`LeftInverseSymplecticAutoencoder`](@ref) is the transpose. A [`BiorthogonalPSDLayer`](@ref) applies two matrices, ``\Phi`` to ``q`` and ``\Psi`` to ``p``, that only have to be *biorthogonal*, ``\Psi^T\Phi = \mathbb{I}_n``. Biorthogonal weights of this kind are used in [otto2023learning, friedl2026learning](@cite). This page shows that the lift is then still symplectic, describes the manifold of biorthogonal pairs as a homogeneous space, and states how the layer parametrizes it.

## Symplecticity and the inverse

For ``\Phi, \Psi\in\mathbb{R}^{N\times{}n}`` let

```math
A = \begin{bmatrix} \Phi & \mathbb{O} \\ \mathbb{O} & \Psi \end{bmatrix}\in\mathbb{R}^{2N\times{}2n}.
```

With ``\mathbb{J}_{2N} = \begin{bmatrix} \mathbb{O} & \mathbb{I}_N \\ -\mathbb{I}_N & \mathbb{O} \end{bmatrix}``,

```math
A^T\mathbb{J}_{2N}A = \begin{bmatrix} \mathbb{O} & \Phi^T\Psi \\ -\Psi^T\Phi & \mathbb{O} \end{bmatrix},
```

so ``A^T\mathbb{J}_{2N}A = \mathbb{J}_{2n}``, i.e. ``A`` is symplectic, exactly when ``\Psi^T\Phi = \mathbb{I}_n``. Its symplectic inverse is

```math
A^+ = \mathbb{J}_{2n}^TA^T\mathbb{J}_{2N} = \begin{bmatrix} \Psi^T & \mathbb{O} \\ \mathbb{O} & \Phi^T \end{bmatrix}, \qquad A^+A = \mathbb{I}_{2n},
```

which is the [`TransposeBiorthogonalPSDLayer`](@ref). The [`PSDLayer`](@ref) is the case ``\Phi = \Psi\in{}St(n, N)``, where ``A^+ = A^T``.

## The manifold of biorthogonal pairs

The pairs form the set

```math
\mathcal{B}(n, N) = \{(\Phi, \Psi)\in\mathbb{R}^{N\times{}n}\times\mathbb{R}^{N\times{}n} : \Psi^T\Phi = \mathbb{I}_n\}.
```

It is a homogeneous space of the general linear group. ``GL(N)`` acts on the product ``\mathbb{R}^{N\times{}n}\times\mathbb{R}^{N\times{}n}`` by

```math
g\cdot(\Phi, \Psi) = (g\Phi, g^{-T}\Psi),
```

which keeps ``\Psi^T\Phi`` fixed, since ``(g^{-T}\Psi)^Tg\Phi = \Psi^Tg^{-1}g\Phi = \Psi^T\Phi``. The action is transitive on ``\mathcal{B}(n, N)``: with ``E = \begin{bmatrix} \mathbb{I}_n \\ \mathbb{O} \end{bmatrix}\in\mathbb{R}^{N\times{}n}`` and a pair ``(\Phi, \Psi)``, let the columns of ``K\in\mathbb{R}^{N\times(N-n)}`` be a basis of the kernel of ``\Psi^T``. The matrix ``G = [\Phi\; K]`` is invertible: if ``\Phi{}a + Kb = 0``, multiplying by ``\Psi^T`` gives ``a = 0``, and then ``b = 0``. Since ``GE = \Phi`` and ``G^T\Psi = \begin{bmatrix} \Phi^T\Psi \\ K^T\Psi \end{bmatrix} = E``,

```math
G\cdot(E, E) = (\Phi, \Psi).
```

The stabilizer of ``(E, E)`` consists of the ``g`` with ``gE = E`` and ``g^TE = E``, the first ``n`` columns and the first ``n`` rows of ``g`` are those of the identity:

```math
g = \begin{bmatrix} \mathbb{I}_n & \mathbb{O} \\ \mathbb{O} & B \end{bmatrix}, \qquad B\in{}GL(N-n).
```

Hence

```math
\mathcal{B}(n, N) \cong GL(N)/GL(N-n), \qquad \dim\mathcal{B}(n, N) = N^2 - (N-n)^2 = 2Nn - n^2,
```

the dimension that `parameterlength` returns for a [`BiorthogonalPSDLayer`](@ref): ``2Nn`` entries and ``n^2`` conditions ``\Psi^T\Phi = \mathbb{I}_n``. This is the counterpart of ``St(n, N)\cong{}O(N)/O(N-n)``: the map ``g\mapsto{}g^{-T}`` fixes exactly ``O(N)``, and the swap ``(\Phi, \Psi)\mapsto(\Psi, \Phi)``, which maps ``\mathcal{B}(n, N)`` to itself, fixes exactly the pairs ``\Phi = \Psi`` with ``\Phi^T\Phi = \mathbb{I}_n``, i.e. ``St(n, N)``.

## The product structure and the parametrization

The two factors of a pair are not independent, but ``\mathcal{B}(n, N)`` is a bundle over the first one. Projecting ``(\Phi, \Psi)\mapsto\Phi`` maps onto the matrices of full rank ``n``, and the fiber over ``\Phi`` is the affine space

```math
\{\Phi(\Phi^T\Phi)^{-1} + \Psi_\perp : \Psi_\perp\in\mathbb{R}^{N\times{}n},\ \Psi_\perp^T\Phi = \mathbb{O}\}
```

of dimension ``(N-n)n``. A matrix of full rank is in turn ``\Phi = UR`` with ``U\in{}St(n, N)`` spanning its columns and ``R\in{}GL(n)``, unique up to ``(U, R)\mapsto(UQ, Q^TR)`` with ``Q\in{}O(n)``. [`BiorthogonalPSDLayer`](@ref) combines the two and stores ``U`` (`weight`), ``R`` (`factor`) and ``W\in\mathbb{R}^{N\times{}n}`` (`complement`):

```math
\Phi = UR, \qquad \Psi = UR^{-T} + (\mathbb{I}_N - UU^T)W.
```

Then ``\Psi^T\Phi = R^{-1}U^TUR + W^T(\mathbb{I}_N - UU^T)UR = \mathbb{I}_n`` for every ``U\in{}St(n, N)``, every invertible ``R`` and every ``W``, and every pair is reached: ``UR^{-T} = \Phi(\Phi^T\Phi)^{-1}``, and ``(\mathbb{I}_N - UU^T)W`` runs through all ``\Psi_\perp``. The parametrization is redundant in the ``O(n)`` above and in the part ``UU^TW`` of ``W``, which does not enter ``\Psi``; the redundancy does not affect the layer. The number of free parameters checks out: ``\dim{}St(n, N) + n^2 + (N-n)n - \dim{}O(n) = 2Nn - n^2``.

Only ``U`` is constrained. It is a [`StiefelManifold`](@extref GeometricOptimizers GeometricOptimizers.StiefelManifold) and is trained by the same manifold optimizers as the weight of a [`PSDLayer`](@ref); ``R`` and ``W`` are trained as Euclidean weights. The layer is initialized with ``R = \mathbb{I}_n`` and ``W = \mathbb{O}``, so that it starts as the [`PSDLayer`](@ref) with weight ``U``, and the layers apply ``R^{-1}`` without forming ``\Phi`` or ``\Psi``. In floating point ``\Psi^T\Phi = \mathbb{I}_n`` holds to round-off times the condition number of ``R``, and the layer requires ``R`` to stay invertible. An optimizer that works on ``GL(N)/GL(N-n)`` directly, through a global tangent space representation as for the Stiefel and Grassmann manifolds, is not implemented.

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
