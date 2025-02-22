```@meta
CurrentModule = GeometricMachineLearning
```

# Geometric Machine Learning

`GeometricMachineLearning` is a package for *structure-preserving scientific machine learning*. It contains models that can learn dynamical systems with geometric structure, such as Hamiltonian (symplectic) or Lagrangian (variational) systems.

In that regard its aim is similar to traditional *geometric numerical integration* [hairer2006geometric, Kraus:2020:GeometricIntegrators](@cite) in that it models maps that share properties with the analytic solution of a differential equation:

![](tikz/gml_venn_light.png)
![](tikz/gml_venn_dark.png)

## Installation

`GeometricMachineLearning` and all of its dependencies can be installed via the Julia REPL by typing 
```julia
]add GeometricMachineLearning
```

## Architectures

Some of the neural network architectures in `GeometricMachineLearning` [brantner2023symplectic, brantner2024volume](@cite) have emerged in connection with developing this package[^1], other have existed before [jin2020sympnets, greydanus2019hamiltonian](@cite).

[^1]: The work on this software package was done in connection with a PhD thesis. You can read its [introduction](@ref "Introduction and Outline") and [conclusion](@ref "Conclusion") here.

New architectures include:
- [symplectic autoencoder](@ref "The Symplectic Autoencoder"),
- [volume-preserving transformers](@ref "Volume-Preserving Transformer"), 
- [linear-symplectic transformer](@ref "Linear Symplectic Transformer"). 

Existing architectures include:
- [SympNets](@ref "SympNet Architecture"),
- [standard transormer](@ref "Standard Transformer").

## Manifolds

`GeometricMachineLearning` supports putting neural network weights on manifolds such as the [Stiefel manifold](@ref "The Stiefel Manifold") and the [Grassmann manifold](@ref "The Grassmann Manifold") and [Riemannian optimization](@ref "Riemannian Manifolds").

![Weights can be put on manifolds to achieve structure preservation or improved stability.](tikz/tangent_vector_light.png)
![Weights can be put on manifolds to achieve structure preservation or improved stability.](tikz/tangent_vector_dark.png)

When `GeometricMachineLearning` optimizes on manifolds it uses the framework introduced in [brantner2023generalizing](@cite). Optimization is necessary for some neural network architectures such as [symplectic autoencoders](@ref "The Symplectic Autoencoder") and can be critical for others such as the [standard transformer](@ref "MNIST Tutorial") [kong2023momentum, zhang2021orthogonality](@cite).



## Special Neural Network Layer

Many layers have been adapted in order to be used for problems in scientific machine learning, such as the [attention layer](@ref "The Attention Layer").

## GPU Support

All neural network layers and architectures that are implemented in `GeometricMachineLearning` have GPU support via the package `KernelAbstractions.jl` [churavy2020kernel](@cite), so `GeometricMachineLearning` naturally integrates `CUDA.jl` [besard2018juliagpu](@cite), `AMDGPU.jl`, `Metal.jl` [besard2022metal](@cite) and `oneAPI.jl` [besard2022one](@cite).

## Tutorials 

There are several tutorials demonstrating how `GeometricMachineLearning` can be used.

These tutorials include:
- a [tutorial on SympNets](@ref "SympNets with `GeometricMachineLearning`") that shows how we can model a flow map corresponding to data coming from an unknown [canonical Hamiltonian system](@ref "Symplectic Systems"),
- a [tutorial on symplectic Autoencoders](@ref "Symplectic Autoencoders and the Toda Lattice") that shows how this architecture can be used in [structure-preserving reduced order modeling](@ref "Hamiltonian Model Order Reduction"),
- a [tutorial on the volume-preserving attention mechanism](@ref "Comparing Different `VolumePreservingAttention` Mechanisms") which serves as a basis for the [volume-preserving transformer](@ref "Volume-Preserving Transformer"),
- a [tutorial on training a transformer with manifold weights for image classification](@ref "MNIST Tutorial") to show that manifold optimization is also useful outside of scientific machine learning.

## Data-Driven Reduced Order Modeling

The main motivation behind developing `GeometricMachineLearning` is *reduced order modeling*, especially *structure-preserving reduced order modeling*. For this purpose we give a short introduction into [this topic](@ref "Basic Concepts of Reduced Order Modeling").