# Proper Orthogonal Decomposition

Proper orthogonal decomposition (POD, [chatterjee2000introduction](@cite)) is perhaps the most widely-used technique for [data-driven reduced order modeling](@ref "Reduced Order Modeling"). POD approximates [the reduction and the reconstruction](@ref "General Workflow") through linear maps. Assume that the big discretized space has dimension ``N`` and we try to model the solution manifold with an ``n``-dimensional subspace. POD then models the reduction ``\mathcal{P}:\mathbb{R}^N\to\mathbb{R}^n`` through a matrix ``\in\mathbb{R}^{n\times{}N}`` and the reconstruction ``\mathcal{R}:\mathbb{R}^n\to\mathbb{R}^N`` through a matrix ``\in\mathbb{R}^{N\times{}n}.`` If we are given a [snapshot matrix](@ref "Snapshot Matrix") finding ``\mathcal{P}`` and ``\mathcal{R}`` amounts to a simple application of *singular value decomposition* (SVD).

```@eval
Main.theorem(raw"Given a snapshot matrix ``M\in\mathbb{R}^{N\times\mathtt{nts}},`` where ``\mathtt{nts}`` is the *number of time steps*, the ideal linear subspace that can best approximate the data stored in ``M`` are the first ``n`` columns of the ``V`` matrix in an SVD ``M = VDU^T.`` The problem of finding this subspace can either be phrased as a maximization problem:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \max_{\psi_1, \ldots, \psi_n\in\mathbb{R}^N} \sum_{i = 1}^n \sum_{j = 1}^{\mathtt{nts}}| \langle u_j, \psi_i \rangle_{\mathbb{R}^N} |^2 \text{ s.t. $\langle \psi_i, \psi_j \rangle = \delta_{ij}$ for $1 \leq i$, $j \leq n$,}
" * Main.indentation * raw"```
" * Main.indentation * raw"or as a minimization problem:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \min_{j = 1}^{\mathtt{nts}} \sum_{j = 1}^{\mathtt{nts}} | u_j - \sum_{i = 1}^n \psi_i\langle u_j, v_i \rangle |^2\text{ s.t. $\langle \psi_i, \psi_j \rangle = \delta_{ij}$ for $1 \leq i$, $j \leq n$.}
" * Main.indentation * raw"```
" * Main.indentation * raw"In both these cases we have 
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \begin{bmatrix} \psi_1 & \psi_2 & \cdots & \psi_n \end{bmatrix} = V\mathtt{[1:N, 1:n]},
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``V`` is obtained via an SVD of ``M``.")
```

A proof of the statement above can be found in e.g. [volkwein2013proper](@cite). We can obtain the reduced equations via [Galerkin projection](@ref "Obtaining the Reduced System via Galerkin Projection"):

```@eval
Main.theorem(raw"Consider a full-order model on ``\mathbb{R}^N`` described by the vector field ``\dot{\hat{u}} = X(\hat{u})``. For a POD basis the reduced vector field, obtained via Galerkin projection, is:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \dot{u} = V^TX(Vu),
" * Main.indentation * raw"```
" * Main.indentation * raw"where we used ``\{\tilde{\psi}_i = Ve_i\}_{i = 1,\ldots, n}`` as test functions. ``e_i\in\mathbb{R}^n`` is the vector that is zero everywhere except for the ``i``-th entry, where it is one.")
``` 

```@eval
Main.proof(raw"If we take as test function ``\tilde{psi}_i = Ve_i``, then we get:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    e_i^TV^TX(Vu(t)) \overset{!}{=} e_i^TV^TV\dot{u}(t) = e_i^T\dot{u}(t),
" * Main.indentation * raw"```
" * Main.indentation * raw"and since this must be true for every ``i = 1, \ldots, n`` we obtain the desired expression for the reduced vector field.")
```

In recent years another approach to model ``\mathcal{P}`` and ``\mathcal{R}`` has become popular, namely to use neural networks to do so.

# Autoencoders

Autoencoders are a popular tool in machine learning to perform *data compression* [goodfellow2016deep](@cite). The idea is always to find a low-dimensional representation of high-dimensional data. This is also referred to as *learning a feature space*. This idea straightforwardly lends itself towards an application in reduced order modeling. In this setting we *learn* two mappings that are modeled with neural networks:

```@eval
Main.definition(raw"An **autoencoder** is a tuple of two mappings ``(\mathcal{P}, \mathcal{R})`` called the **reduction** and the **reconstruction**:
" * Main.indentation * raw"1. The reduction ``\mathcal{P}:\mathbb{R}^N\to\mathbb{R}^n`` is modeled with a neural network that maps high-dimensional data to a low-dimensional feature space. This network is also referred to as the **encoder** and we routinely denote it by ``\Psi^\mathrm{enc}_{\theta_1}`` to stress the parameter-dependence on ``\theta_1``.
" * Main.indentation * raw"2. The reconstruction ``\mathcal{R}:\mathbb{R}^n\to\mathbb{R}^N`` is modeled with a neural network that maps inputs from the low-dimensional feature space to the high-dimensional space in which the original data were collected. This network is also referred to as the **decoder** and we routinely denote it by ``\Psi^\mathrm{dec}_{\theta_2}`` to stress the parameter-dependence on ``\theta_2``.
" * Main.indentation * raw"During training we optimize the autoencoder for minimizing the *projection error*.")
```

Unlike in the POD case we have to resort to using [neural network optimizers](@ref " Neural Network Optimizers") in order to adapt the neural network to the data at hand as opposed to simply using SVD. The use of autoencoders instead of POD is extremely advantageous in the case when we deal with problems that exhibit a slowly-decaying [Kolmogorov ``n``-width](@ref). During training we minimize the [projection error](@ref "Projection Error").

```@eval
Main.remark(raw"Note that POD can be seen as a special case of an autoencoder where the encoder and the decoder both consist of only one matrix. If we restrict this matrix to be orthonormal, i.e. optimize on the Stiefel manifold, then the best solution we can obtain is equivalent to applying SVD and finding the POD basis.")
```

## The Reduced Equations for the Autoencoder




Both POD and standard autoencoders suffer from the problem that they completely neglect the structure of the differential equation and the data they are applied to. This can have grave consequences [peng2016symplectic, tyranowski2023symplectic, buchfink2023symplectic](@cite).

## Library Functions

```@docs
GeometricMachineLearning.AutoEncoder
GeometricMachineLearning.Encoder
GeometricMachineLearning.Decoder
GeometricMachineLearning.UnknownEncoder
GeometricMachineLearning.UnknownDecoder
encoder
decoder
```

## References

```@bibliography
Canonical = false
Pages = []

chatterjee2000introduction
```