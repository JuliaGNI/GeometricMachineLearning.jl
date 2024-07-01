# Symplectic Autoencoder 

Symplectic autoencoders offer a structure-preserving way of mapping a high-dimensional system to a low dimensional system. Concretely this means that if we obtain a reduced system by means of a symplectic autoencoder, this system will again be reduced. 

The architecture is represented by the figure below:

```@example 
Main.include_graphics("../tikz/symplectic_autoencoder") # hide
```

It is a composition of [SympNet gradient layers](@ref "SympNet Gradient Layer") and PSD-like matrices.


## Intermediate Dimensions

For a high-fidelity system of dimension ``2N`` and a reduced system of dimension ``2n``, the intermediate dimensions in the symplectic encoder and the decoder are computed according to: 

```julia
iterations = Vector{Int}(n : (N - n) ÷ (n_blocks - 1) : N)
iterations[end] = full_dim2
iterations * 2
```

So for e.g. ``2N = 100,`` ``2n = 10`` and ``\mathtt{n\_blocks} = 3`` we get 

```math
\mathrm{iterations} = 5\mathtt{:}(45 \div 2)\mathtt{:}50 = 5\mathtt{:}22\mathtt{:}50 = (5, 27, 49),
```

and after the further two modifications the dimensions are:

```math
(10, 54, 100).
```


## Example

A visualization of an instance of [SymplecticAutoencoder](@ref) is shown below: 

```@example 
Main.include_graphics("../tikz/symplectic_autoencoder_architecture") # hide
```

In this example shown in the figure `n_encoder_blocks` is two, `n_encoder_layers` is four, `n_decoder_blocks` is 3 and `n_decoder_layers` is 2. You can build such an instance of a symplectic autoencoder by calling:

```@example sae
using GeometricMachineLearning

const full_dim = 100
const reduced_dim = 10

model = SymplecticAutoencoder(full_dim, reduced_dim; n_encoder_blocks = 2, n_encoder_layers = 4, n_decoder_blocks = 3, n_decoder_layers = 2)

for layer in Chain(model)
    println(stdout, layer)
end
```

We also see that the intermediate dimension in the decoder is 54.

## Library Functions

```@docs; canonical = false
SymplecticAutoencoder
```

## References

```@bibliography
Pages = []
Canonical = false

brantner2023symplectic
```