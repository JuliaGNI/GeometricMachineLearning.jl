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

**Little of that is tested.** `test/metal/` runs two checks on an Apple GPU, on every Apple-silicon Mac and in the macOS jobs of CI: the products of a `PoissonTensor` with a wrapped device array, and the `tensor_mat_mul!` kernel. A green test matrix says nothing about the rest of the GPU path. What is written below was measured by hand on an Apple M4 Max through `Metal.jl`, in `Float32`, because Apple GPUs have no `Float64` at all, and against `GeometricOptimizers` 0.8.0.

Everything tried on that device ran. Constructed, and then applied to both a matrix and a 3-tensor: `GSympNet`, `LASympNet`, `StandardTransformerIntegrator`, `LinearSymplecticTransformer`, `SymplecticTransformer` at both its default `transformer_dim` and an upscaling one, `VolumePreservingFeedForward`, `VolumePreservingTransformer`, `SymplecticAutoencoder`, `PSDArch` and `Transformer(…; Stiefel = true)`. `ClassificationTransformer` was constructed but not applied, because its input is an image. The tensor kernels and `map_to_cpu` run, and so does training with `Optimizer`, which returns a `Float32` history and leaves the parameters on the device — for `PSDArch` that includes the manifold weights, which stay a `StiefelManifold` over an `MtlMatrix`.

Two of those results rest on `GeometricOptimizers` rather than on anything here.

- `VolumePreservingFeedForward` and `VolumePreservingTransformer` applied to a **matrix** multiply a `LowerTriangular` or `UpperTriangular` weight by that matrix. `GeometricOptimizers` supplies a `KernelAbstractions` kernel for that product, so the call does not fall through to a generic multiply that reads one entry at a time, which `GPUArraysCore` refuses.
- `StiefelLayer`, `GrassmannLayer` and `PSDLayer`, and so every architecture that holds one of them, orthonormalize their weight at **construction** through `GeometricOptimizers.orthonormal_columns`. That is CholeskyQR2 — matrix products and triangular solves only — so it runs wherever the draw was allocated. `LinearAlgebra.qr!` is a host factorization and `Metal.jl` implements no `qr` for an `MtlArray`, so a device needs the other one.

## Tutorials 

There are several tutorials demonstrating how `GeometricMachineLearning` can be used.

These tutorials include:
- a [tutorial on SympNets](@ref "SympNets with `GeometricMachineLearning`") that shows how we can model a flow map corresponding to data coming from an unknown [canonical Hamiltonian system](@ref "Symplectic Systems"),
- a [tutorial on symplectic Autoencoders](@ref "Symplectic Autoencoders and the Toda Lattice") that shows how this architecture can be used in [structure-preserving reduced order modeling](@ref "Hamiltonian Model Order Reduction"),
- a [tutorial on the volume-preserving attention mechanism](@ref "Comparing Different `VolumePreservingAttention` Mechanisms") which serves as a basis for the [volume-preserving transformer](@ref "Volume-Preserving Transformer"),
- a [tutorial on training a transformer with manifold weights for image classification](https://juliagni.github.io/GMLDatasets.jl/latest/mnist/mnist_tutorial/), in the companion package `GMLDatasets`, to show that manifold optimization is also useful outside of scientific machine learning.

## Data-Driven Reduced Order Modeling

The main motivation behind developing `GeometricMachineLearning` is *reduced order modeling*, especially *structure-preserving reduced order modeling*. For this purpose we give a short introduction into [this topic](@ref "Basic Concepts of Reduced Order Modeling").