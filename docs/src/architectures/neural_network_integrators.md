# Neural Network Integrators 

In `GeometricMachineLearning` we can divide most neural network architectures (that are used for applications to physical systems) into two categories: autoencoders and integrators. *Integrator* in its most general form refers to an approximation of the flow of an ODE (see [the section on the existence and uniqueness theorem](@ref eau_th)) by a numerical scheme. Traditionally these numerical schemes were constructed by defining certain relationships between a known time step ``z^{(t)}`` and a future unknown one ``z^{(t+1)}`` [hairer2006geometric, leimkuhler2004simulating](@cite): 

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

[`TransformerIntegrator`](@ref)


```@bibliography
Pages = []
Canonical = false

hairer2006geometric
leimkuhler2004simulating
Kraus:2020:GeometricIntegrators
feng1998step
```

```@docs; canonical=false
NeuralNetworkIntegrator 
TransformerIntegrator
```