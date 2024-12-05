# Hamiltonian Neural Network

The Hamiltonian Neural Network (HNN) [greydanus2019hamiltonian](@cite) aims at building a [Hamiltonian vector field](@ref "Symplectic Systems")  with a neural network. We recall that a *canonical Hamiltonian vector field* on ``\mathbb{R}^{2d}`` is one that can be written as:

```math
    X_H(z) = \mathbb{J}_{2d}\nabla_zH,
```
where ``\mathbb{J}_{2d}`` is the [`PoissonTensor`](@ref). The idea behind a Hamiltonian neural network is to learn a vector field of this form, i.e. to learn:

```math
    X_{\mathcal{NN}}(z) = \mathbb{J}_{2d}\nabla_z\mathcal{NN},
```
where ``\mathcal{NN}:\mathbb{R}^{2d}\to\mathbb{R}`` is a neural network that approximates the Hamiltonian. There are then two different options to define a *HNN loss*, depending on the format in which the data are given.

## HNN Loss for Vector Field Data

For the first loss, we assume that the given data describe the vector field of the HNN. 

## HNN Loss for Phase Space Data

For the second loss, we assume that the given data describe points in phase space associated to a Hamiltonian system. 


!!! info
   Usually we use [`Zygote`](https://github.com/FluxML/Zygote.jl) for computing derivatives in `GeometricMachineLearning`, but as the [`Zygote` documentation](https://fluxml.ai/Zygote.jl/dev/limitations/#Second-derivatives-1) itself points out: "Often using a different AD system over Zygote is a better solution [for computing second-order derivatives]." For this reason we compute the loss of the HNN with [`SymbolicNeuralNetworks`](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl) and optionally also its gradient.