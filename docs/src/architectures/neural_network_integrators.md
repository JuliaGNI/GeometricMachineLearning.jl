# Neural Network Integrators 

In `GeometricMachineLearning` we can divide most neural network architectures (that are used for applications to physical systems) into two categories: autoencoders and integrators. This is also closely related to the application of reduced order modeling where *autoencoders are used in the offline phase* and *integrators are used in the online phase*.

The term *integrator* in its most general form refers to an approximation of the [flow of an ODE](@ref "The Existence-And-Uniqueness Theorem") by a numerical scheme. Traditionally, for so called *one-step methods*, these numerical schemes are constructed by defining certain relationships between a known time step ``z^{(t)}`` and a future unknown one ``z^{(t+1)}`` [hairer2006geometric, leimkuhler2004simulating](@cite): 

```math
    f(z^{(t)}, z^{(t+1)}) = 0.
```

One usually refers to such a relationship as an *integration scheme*. If this relationship can be reformulated as 

```math
    z^{(t+1)} = g(z^{(t)}),
```

then we refer to the scheme as *explicit*, if it cannot be reformulated in such a way then we refer to it as *implicit*. Implicit schemes are typically more expensive to solve than explicit ones. The `Julia` library `GeometricIntegrators` [Kraus:2020:GeometricIntegrators](@cite) offers a wide variety of integration schemes both implicit and explicit. 

The neural network integrators in `GeometricMachineLearning` (the corresponding type is [`NeuralNetworkIntegrator`](@ref)) are all explicit integration schemes where the function ``g`` above is modeled with a neural network.

Neural networks, as an alternative to traditional methods, are employed because of (i) potentially superior performance and (ii) an ability to learn unknown dynamics from data. 

The simplest of such a neural network for modeling an explicit integrator is the [`ResNet`](@ref). [SympNets](@ref "SympNet Architecture") can be seen as the [symplectic](@ref "Symplectic Systems") version of the ResNet. There is an example [demonstrating the performance of SympNets](@ref "SympNets with `GeometricMachineLearning`"). This example demonstrates the advantages of symplectic neural networks.

## Multi-step methods

*Multi-step method* [feng1987symplectic, ge1988approximation](@cite) refers to schemes that are of the form[^1]: 

[^1]: We again assume that all the steps up to and including ``t`` are known.

```math
    f(z^{(t - \mathtt{sl} + 1)}, z^{(t - \mathtt{sl} + 2)}, \ldots, z^{(t)}, z^{(t + 1)}, \ldots, z^{(\mathtt{pw} + 1)}) = 0,
```
where `sl` is short for *sequence length* and `pw` is short for *prediction window*. Note that we can recover traditional one-step methods by setting `sl` and `pw` equal to 1. We can also formulate explicit mulit-step methods. They are of the form: 

```math 
[z^{(t+1)}, \ldots, z^{(t+\mathtt{pw})}] = g(z^{(t - \mathtt{sl} + 1)}, \ldots, z^{(t)}).
```

In `GeometricMachineLearning` all multi-step methods, as is the case with one-step methods, are explicit. There are essentially two ways to construct multi-step methods with neural networks: the older one is using recurrent neural networks such as long short-term memory cells (LSTMs) [hochreiter1997long](@cite) and the newer one is using transformer neural networks [vaswani2017attention](@cite). Both of these approaches have been successfully employed to learn multi-step methods (see [fresca2021comprehensive, lee2020model](@cite) for the former and [hemmasian2023reduced, solera2023beta, brantner2024volume](@cite) for the latter), but because the transformer architecture exhibits superior performance on modern hardware and can be imbued with geometric properties we almost always use a transformer-derived architecture when dealing with time series[^2].

[^2]: `GeometricMachineLearning` also has an LSTM implementation, but this may be deprecated in the future. 

Explicit multi-step methods derived from the transformer are always subtypes of the type [`TransformerIntegrator`](@ref) in `GeometricMachineLearning`. In `GeometricMachineLearning` the [standard transformer](@ref "Standard Transformer"), the [volume-preserving transformer](@ref "Volume-Preserving Transformer") and the [linear symplectic transformer](@ref "Linear Symplectic Transformer") are implemented. 

```@eval
Main.remark(raw"For standard multi-step methods (that are not neural network-based) `sl` is generally a number greater than one whereas `pw = 1` in most cases. 
" * Main.indentation * raw"For the `TransformerIntegrator`s in `GeometricMachineLearning` however we usually have:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathtt{pw} = \mathtt{sl},
" * Main.indentation * raw"```
" * Main.indentation * raw"so the number of vectors in the input sequence is equal to the number of vectors in the output sequence. This makes it easier to define structure-preservation for these architectures and improves stability.")
```

## Library Functions 

```@docs
NeuralNetworkIntegrator
ResNet
GeometricMachineLearning.ResNetLayer
iterate(::NeuralNetwork{<:NeuralNetworkIntegrator}, ::BT) where {T, AT<:AbstractVector{T}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}}
TransformerIntegrator
iterate(::NeuralNetwork{<:TransformerIntegrator}, ::NamedTuple{(:q, :p), Tuple{AT, AT}}) where {T, AT<:AbstractMatrix{T}}
```

```@raw latex
\begin{comment}
```

## References

```@bibliography
Pages = []
Canonical = false

hairer2006geometric
leimkuhler2004simulating
Kraus:2020:GeometricIntegrators
feng1998step
vaswani2017attention
hemmasian2023reduced
solera2023beta
brantner2024volume
```

```@raw latex
\end{comment}
```