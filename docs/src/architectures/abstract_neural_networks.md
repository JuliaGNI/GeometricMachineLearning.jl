```@raw latex
In this chapter we build, starting from the neural network layers introduced in the previous chapter, \textit{neural network architectures}. Here these are always understood as a composition of neural network layers. We start by shortly explaining the application interface for neural networks used in \texttt{GeometricMachineLearning} and then introduce symplectic autoencoders, SympNets, volume-preserving feedforward neural networks, standard transformers, volume-preserving transformers and linear symplectic transformers. All of these architectures, except SympNets and standard transformers (and arguably volume-preserving feedforward neural networks), constitute new work. All of them, except symplectic autoencoders, can be seen as \textit{neural network-based integrators}.
```

# `NeuralNetwork`s in `GeometricMachineLearning`

`GeometricMachineLearning` inherits some functionality from another `Julia` package called [`AbstractNeuralNetworks`](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl). How these two packages interact is shown in the figure below for the example of the [SympNet](@ref "SympNet Architecture")[^1]:

[^1]: The section on SympNets also contains an explanation of all the `struct`s and `type`s described in this section here.

```@example 
Main.include_graphics("../tikz/structs_visualization"; width = .99, caption = raw"Visualization of how the packages interact. ") # hide
```

The red color indicates an `abstract type`, blue indicates a `struct` and orange indicates a `const` (derived from a `struct`). Solid black arrows indicate direct dependencies, i.e. we have

```@example abstract_neural_networks
using GeometricMachineLearning # hide
using GeometricMachineLearning: GradientLayer, Architecture, SympNetLayer, AbstractExplicitLayer # hide
@assert GradientLayer <: SympNetLayer <: AbstractExplicitLayer # hide
GradientLayer <: SympNetLayer <: AbstractExplicitLayer
@assert GSympNet <: SympNet <: Architecture # hide
GSympNet <: SympNet <: Architecture
nothing # hide
```

Dashed black arrows indicate a derived neural network architecture. A [`GSympNet`](@ref) (which is an `Architecture`) is derived from [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref) (which are `AbstractExplicitLayer`s) for example. An `Architecture` can be turned into a `NeuralNetwork` by calling the associated constructor

```@example abstract_neural_networks
arch = GSympNet(3)
nn = NeuralNetwork(arch, CPU(), Float64)
nothing # hide
```

Such a neural network has four fields:
- `architecture`: the `Architecture` we supplied the constructor with,
- `model`: a *translation* of the supplied architecture into specific neural network layers,
- `params`: the neural network parameters,
- `backend`: this indicates on which *device* we allocate the neural network parameters. In this case it is `CPU()`.

We can get the associated model to `GSympNet` by calling:

```@example abstract_neural_networks
nn.model.layers
```

and we see that it consists of two layers: a [`GradientLayerQ`](@ref) and a [`GradientLayerP`](@ref).