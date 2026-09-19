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

Some of the neural network architectures in `GeometricMachineLearning` [brantner2023symplectic, brantner2025volume](@cite) have emerged in connection with developing this package[^1], other have existed before [jin2020sympnets, greydanus2019hamiltonian](@cite).

[^1]: The work on this software package was done in connection with a PhD thesis. You can read its [introduction](@ref "Introduction and Outline") and [conclusion](@ref "Conclusion") here.

New architectures include:
- [symplectic autoencoder](@ref "The Symplectic Autoencoder"),
- [volume-preserving transformers](@ref "Volume-Preserving Transformer"), 
- [linear-symplectic transformer](@ref "Linear Symplectic Transformer"). 

Existing architectures include:
- [SympNets](@ref "SympNet Architecture"),
- [standard transormer](@ref "Standard Transformer").

## Manifolds

`GeometricMachineLearning` supports putting neural network weights on manifolds such as the [Stiefel manifold](@extref GeometricOptimizers The-Stiefel-Manifold) and the [Grassmann manifold](@extref GeometricOptimizers The-Grassmann-Manifold) and [Riemannian optimization](@extref GeometricOptimizers Riemannian-Manifolds).

![Weights can be put on manifolds to achieve structure preservation or improved stability.](tikz/tangent_vector_light.png)
![Weights can be put on manifolds to achieve structure preservation or improved stability.](tikz/tangent_vector_dark.png)

When `GeometricMachineLearning` optimizes on manifolds it uses the framework introduced in [brantner2023generalizing](@cite). Optimization is necessary for some neural network architectures such as [symplectic autoencoders](@ref "The Symplectic Autoencoder") and can be critical for others such as the [standard transformer](https://juliagni.github.io/GMLDatasets.jl/latest/mnist/mnist_tutorial/) [kong2023momentum, zhang2021orthogonality](@cite).



## Special Neural Network Layer

Many layers have been adapted in order to be used for problems in scientific machine learning, such as the [attention layer](@ref "The Attention Layer").

## GPU Support

`GeometricMachineLearning` allocates and computes through `KernelAbstractions.jl` [churavy2020kernel](@cite), so its layers and architectures are written against any backend that package supports: `CUDA.jl` [besard2018juliagpu](@cite), `AMDGPU.jl`, `Metal.jl` [besard2022metal](@cite) and `oneAPI.jl` [besard2022one](@cite).

**None of that is tested.** There is no GPU test under `test/` and no GPU job in CI, so a green test matrix says nothing about the GPU path. What is written below was measured by hand on an Apple M4 Max through `Metal.jl`, in `Float32`, because Apple GPUs have no `Float64` at all.

What ran on that device: the tensor kernels and `map_to_cpu`; the forward pass of `GSympNet`, `LASympNet`, `StandardTransformerIntegrator`, `LinearSymplecticTransformer` and `SymplecticTransformer`, on both matrix and tensor input; the forward pass of `VolumePreservingFeedForward` and `VolumePreservingTransformer` on tensor input; and training a network with `Optimizer`, which returns a `Float32` history and leaves the parameters on the device.

Two things did not run.

- `VolumePreservingFeedForward` and `VolumePreservingTransformer` applied to a **matrix** raise *Scalar indexing is disallowed*. The product of a structured matrix by a matrix belongs to `GeometricOptimizers`, and the guard belongs to `GPUArraysCore`, which `CUDA.jl`, `AMDGPU.jl` and `oneAPI.jl` share. The tensor path is unaffected.
- The manifold layers — `StiefelLayer`, `GrassmannLayer` and `PSDLayer`, and so `SymplecticAutoencoder`, `PSDArch` and `MultiHeadAttention(…; Stiefel = true)` — fail at **construction**. They orthonormalize their weight with `LinearAlgebra.qr!`, which is a host factorization; `Metal.jl` implements no `qr` for an `MtlArray`.

## Tutorials 

There are several tutorials demonstrating how `GeometricMachineLearning` can be used.

These tutorials include:
- a [tutorial on SympNets](@ref "SympNets with `GeometricMachineLearning`") that shows how we can model a flow map corresponding to data coming from an unknown [canonical Hamiltonian system](@ref "Symplectic Systems"),
- a [tutorial on symplectic Autoencoders](@ref "Symplectic Autoencoders and the Toda Lattice") that shows how this architecture can be used in [structure-preserving reduced order modeling](@ref "Hamiltonian Model Order Reduction"),
- a [tutorial on the volume-preserving attention mechanism](@ref "Comparing Different `VolumePreservingAttention` Mechanisms") which serves as a basis for the [volume-preserving transformer](@ref "Volume-Preserving Transformer"),
- a [tutorial on training a transformer with manifold weights for image classification](https://juliagni.github.io/GMLDatasets.jl/latest/mnist/mnist_tutorial/), in the companion package `GMLDatasets`, to show that manifold optimization is also useful outside of scientific machine learning.

## Data-Driven Reduced Order Modeling

The main motivation behind developing `GeometricMachineLearning` is *reduced order modeling*, especially *structure-preserving reduced order modeling*. For this purpose we give a short introduction into [this topic](@ref "Basic Concepts of Reduced Order Modeling").