```@meta
CurrentModule = GeometricMachineLearning
```

# Geometric Machine Learning

GeometricMachineLearning.jl implements various scientific machine learning models that aim at learning dynamical systems with geometric structure, such as Hamiltonian (symplectic) or Lagrangian (variational) systems.

## Installation

*GeometricMachineLearning.jl* and all of its dependencies can be installed via the Julia REPL by typing 
```julia
]add GeometricMachineLearning
```

## Architectures

There are several architectures tailored towards problems in scientific machine learning implemented in `GeometricMachineLearning`.

```@contents
Pages = [
    "architectures/sympnet.md",
]
```

## Manifolds

`GeometricMachineLearning` supports putting neural network weights on manifolds. These include:

```@contents
Pages = [
    "manifolds/grassmann_manifold.md",
    "manifolds/stiefel_manifold.md",
]
```

## Special Neural Network Layer

Many layers have been adapted in order to be used for problems in scientific machine learning. Including:

```@contents
Pages = [
    "layers/attention_layer.md",
]
```

## Tutorials 

Tutorials for using `GeometricMachineLearning` are: 

```@contents
Pages = [
    "tutorials/sympnet_tutorial.md",
    "tutorials/mnist_tutorial.md",
]
```

## Reduced Order Modeling

A short description of the key concepts in **reduced order modeling** (where `GeometricMachineLearning` can be used) are in:

```@contents
Pages = [
    "reduced_order_modeling/autoencoder.md",
    "reduced_order_modeling/symplectic_autoencoder.md",
    "reduced_order_modeling/kolmogorov_n_width.md",
]
```