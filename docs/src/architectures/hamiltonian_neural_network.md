# [Hamiltonian Neural Network](@id hnn_architecture)

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

For the first loss, we assume that the given data describe the vector field of the HNN:

```math
\mathcal{L}_\mathrm{HNN} = \sqrt{\sum_{i=1}^d\left(\Big{||}\frac{\partial\mathcal{NN}}{\partial{}q_i} + \dot{p}_i \Big{||}_2^2 + \Big{||} \frac{\partial\mathcal{NN}}{\partial{}p_i} - \dot{q}_i\Big{||}_2^2\right)}
```

## HNN Loss for Phase Space Data

For the second loss, we assume that the given data describe points in phase space associated to a Hamiltonian system. For this approach we also need to specify a *symplectic integrator* [hairer2006geometric](@cite) in order to train the neural network. In the following we use [SymplecticEulerB](https://juliagni.github.io/GeometricIntegrators.jl/latest/modules/integrators/#GeometricIntegrators.Integrators.SymplecticEulerB) to define this loss. This integrator does the following:

```math
\mathrm{SymplecticEulerB}: (q^{(t)}, p^{(t)}) \mapsto (q^{(t+1)}, p^{(t+1)})
```

Note that this integrator is implicit in general.

```math
\mathcal{L}_\mathrm{HNN} = \sqrt{sum_t\sum_{i=1}^d \Big{||} \frac{\partial\mathcal{NN}}{\partial{}q_i}(q^{(t)}, p^{(t+1)}) + \frac{p_i^{t+1} - p_i^{(t)}}{\Delta{}t} \Big{||}_2^2 + \Big{||} \frac{\partial\mathcal{NN}}{\partial{}p_i}(q^{(t)}, p^{(t+1)}) + \frac{q_i^{t+1} - q_i^{(t)}}{\Delta{}t} \Big{||}_2^2}
```

Here the derivatives (i.e. vector field data) ``\dot{q}_i^{(t)}`` and ``\dot{p}_i^{(t)}`` are approximated with finite differences: 

!!! info
   Usually we use [`Zygote`](https://github.com/FluxML/Zygote.jl) for computing derivatives in `GeometricMachineLearning`, but as the [`Zygote` documentation](https://fluxml.ai/Zygote.jl/dev/limitations/#Second-derivatives-1) itself points out: "Often using a different AD system over Zygote is a better solution [for computing second-order derivatives]." For this reason we compute the loss of the HNN with [`SymbolicNeuralNetworks`](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl) and optionally also its gradient.

## Library Functions

```@docs
GeometricMachineLearning.hamiltonian_vector_field(::HamiltonianArchitecture)
GeometricMachineLearning.HamiltonianArchitecture
GeometricMachineLearning.StandardHamiltonianArchitecture
GeometricMachineLearning.HNNLoss
GeometricMachineLearning.symbolic_hamiltonian_vector_field(::GeometricMachineLearning.SymbolicNeuralNetwork)
GeometricMachineLearning.SymbolicPullback(::HamiltonianArchitecture)
GeometricMachineLearning.GeneralizedHamiltonianArchitecture
GeometricMachineLearning._processing
```