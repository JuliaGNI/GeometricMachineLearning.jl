# Neural Network Integrators 

In `GeometricMachineLearning` we can divide most neural network architectures (that are used for applications to physical systems) into two categories: autoencoders and integrators. *Integrator* in its most general form refers to an approximation of the flow of an ODE (see [the section on the existence and uniqueness theorem](@ref "The Existence-And-Uniqueness Theorem")) by a numerical scheme. Traditionally these numerical schemes were constructed by defining certain relationships between a known time step ``z^{(t)}`` and a future unknown one ``z^{(t+1)}`` [hairer2006geometric, leimkuhler2004simulating](@cite): 

```math
    f(z^{(t)}, z^{(t+1)}) = 0.
```

One usually refers to such a relationship as an "integration scheme". If this relationship can be reformulated as 

```math
    z^{(t+1)} = g(z^{(t)}),
```

then we refer to the scheme as *explicit*, if it cannot be reformulated in such a way then we refer to it as *implicit*. Implicit schemes are typically more expensive to solve than explicit ones. The `Julia` library `GeometricIntegrators` [Kraus:2020:GeometricIntegrators](@cite) offers a wide variety of integration schemes both implicit and explicit. 

The neural network integrators in `GeometricMachineLearning` (the corresponding type is [`NeuralNetworkIntegrator`](@ref)) are all explicit integration schemes where the function ``g`` above is modeled with a neural network.

Neural networks, as an alternative to traditional methods, are employed because of (i) potentially superior performance and (ii) an ability to learn unknown dynamics from data. 

## Multi-step methods

*Multi-step method* [feng1987symplectic, ge1988approximation](@cite) refers to schemes that are of the form[^1]: 

[^1]: We again assume that all the steps up to and including ``t`` are known.

```math
    f(z^{(t - \mathtt{sl} + 1)}, z^{(t - \mathtt{sl} + 2)}, \ldots, z^{(t)}, z^{(t + 1)}, \ldots, z^{(\mathtt{pw} + 1)}) = 0,
```
where `sl` is short for *sequence length* and `pw` is short for *prediction window*. In contrast to traditional single-step methods, `sl` and `pw` can be greater than 1. An explicit multi-step method has the following form: 

```math 
[z^{(t+1)}, \ldots, z^{(t+\mathtt{pw})}] = g(z^{(t - \mathtt{sl} + 1)}, \ldots, z^{(t)}).
```

There are essentially two ways to construct multi-step methods with neural networks: the older one is using recurrent neural networks such as long short-term memory cells (LSTMs, [hochreiter1997long](@cite)) and the newer one is using transformer neural networks [vaswani2017attention](@cite). Both of these approaches have been successfully employed to learn multi-step methods (see [fresca2021comprehensive, lee2020model](@cite) for the former and [hemmasian2023reduced, solera2023beta, brantner2024volume](@cite) for the latter), but because the transformer architecture exhibits superior performance on modern hardware and can be imbued with geometric properties it is recommended to always use a transformer-derived architecture when dealing with time series[^2].

[^2]: `GeometricMachineLearning` also has an LSTM implementation, but this may be deprecated in the future. 

Explicit multi-step methods derived from he transformer are always subtypes of the type [`TransformerIntegrator`](@ref) in `GeometricMachineLearning`. In `GeometricMachineLearning` the [standard transformer](@ref "Standard Transformer"), the [volume-preserving transformer](@ref "Volume-Preserving Transformer") and the [linear symplectic transformer](@ref "Linear Symplectic Transformer") are implemented. 

## Library Functions 

```@docs; canonical=false
NeuralNetworkIntegrator 
TransformerIntegrator
```

## References 

```@bibliography
Pages = []
Canonical = false

hairer2006geometric
leimkuhler2004simulating
Kraus:2020:GeometricIntegrators
feng1998step
hochreiter1997long
vaswani2017attention
fresca2021comprehensive
lee2020model
hemmasian2023reduced
solera2023beta
brantner2024volume
```