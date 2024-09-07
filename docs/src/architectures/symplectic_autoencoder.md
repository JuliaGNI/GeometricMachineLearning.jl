# The Symplectic Autoencoder 

Symplectic autoencoders offer a structure-preserving way of mapping a high-dimensional system to a low-dimensional system. Concretely this means that if we obtain a reduced system by means of a symplectic autoencoder, this system will again be symplectic; we can thus model a symplectic FOM with a [symplectic ROM](@ref "The Symplectic Solution Manifold"). 

The architecture is represented by the figure below[^1]:

[^1]: For the symplectic autoencoder we only use [SympNet gradient layers](@ref "SympNet Gradient Layer") because they seem to outperform ``LA``-SympNets in many cases and are easier to interpret: their nonlinear part is the gradient of a function that only depends on half the coordinates.

```@example 
Main.include_graphics("../tikz/symplectic_autoencoder"; width = .7, caption = raw"A visualization of the symplectic autoencoder architecture. It is a composition of SympNet layers and PSD-like layers.") # hide
```

It is a composition of [SympNet gradient layers](@ref "SympNet Gradient Layer") and [PSD-like matrices](@ref "Proper Symplectic Decomposition"), so a matrix ``A_i`` (respectively ``A_i^+``) is of the form

```math
    A_i^{(+)} = \begin{bmatrix} \Phi_i & \mathbb{O} \\ \mathbb{O} & \Phi_i \end{bmatrix} \text{ where }\begin{cases} \Phi_i\in{}St(d_{i},d_{i+1})\subset\mathbb{R}^{d_{i+1}\times{}d_i} & \text{if $d_{i+1} > d_i$}
    \\
    \Phi_i\in{}St(d_{i+1},d_{i})\subset\mathbb{R}^{d{i}\times{}d_{i+1}} & \text{if $d_i > d_{i+1}$},
    \end{cases}
```

where ``A_i^{(+)} = A_i`` if ``d_{i+1} > d_i`` and ``A_i^{(+)} = A_i^+`` if ``d_{i+1} < d_i.`` Also note that for cotangent lift-like matrices we have

```math
\begin{aligned}
    A_i^+ = \mathbb{J}_{2N} A_i^T \mathbb{J}_{2n}^T & = \begin{bmatrix} \mathbb{O}_{n\times{}n} & \mathbb{I}_n \\ -\mathbb{I}_n & \mathbb{O}_{n\times{}n} \end{bmatrix} \begin{bmatrix} \Phi_i^T & \mathbb{O}_{n\times{}N} \\ \mathbb{O}_{n\times{}N} & \Phi_i^T \end{bmatrix} \begin{bmatrix} \mathbb{O}_{N\times{}N} & - \mathbb{I}_N \\ \mathbb{I}_N & \mathbb{O}_{N\times{}N} \end{bmatrix} \\ & = \begin{bmatrix} \Phi_i^T & \mathbb{O}_{n\times{}N} \\ \mathbb{O}_{n\times{}N} & \Phi_i^T \end{bmatrix} = A_i^T,
\end{aligned}
```

so the symplectic inverse is equivalent to a matrix transpose in this case. In the symplectic autoencoder we use SympNets as a form of *symplectic preprocessing* before the linear symplectic reduction (i.e. the PSD layer) is employed. The resulting neural network has some of its weights on manifolds, which is why we cannot use standard neural network optimizers, but have to resort to [manifold optimizers](@ref "Generalization to Homogeneous Spaces"). Note that manifold optimization is not necessary for the weights corresponding to the SympNet layers, these are still updated with standard neural network optimizers during training. Also note that SympNets are nonlinear and preserve symplecticity, but they cannot change the dimension of a system while PSD layers can change the dimension of a system and preserve symplecticity, but are strictly linear. Symplectic autoencoders have all three properties: they preserve symplecticity, can change dimension and are nonlinear mappings. We can visualize this in a Venn diagram:

```@example
Main.include_graphics("../tikz/sae_venn"; caption = raw"Venn diagram visualizing that a symplectic autoencoder (SAE) is symplectic, can change dimension and is nonlinear. ") # hide
```

We now show the proof that shows ``\nabla_{\mathcal{R}(z)}\psi = (\nabla_{z}\mathcal{R})^+`` which was used when [showing the equivalence between Hamiltonian systems on the full and the reduced space](@ref "The Symplectic Solution Manifold"):
```@eval
Main.proof(raw"""The symplectic autoencoder is a composition of ``G``-SympNet layers and PSD-like matrices:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\Psi^d = A_n\circ\psi_n\circ\cdots\circ{}A_1\circ\psi_1.
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""It's local inverse is
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""(\Psi^d)^{-1} = \psi_1^{-1}\circ{}A_1^+\circ\ldots\circ\psi_n^{-1}\circ{}A_n^+.
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""The jacobian of ``\Psi^d`` is:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\nabla_z\Psi^d = A_n\nabla_{A_{n-1}\cdots{}A_1\psi_1(z)}\psi_n\cdots{}A_1\nabla_z\psi_1,
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""and thus
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""(\nabla_z\Psi^d)^+ = (\nabla\psi)^+A_1^+\cdots(\nabla\psi_n)^+A_n^+,
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""where we dropped the argument in the derivative of the nonlinear parts. We further have
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""A^+ = A^T
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""for PSD-like matrices and
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""(\nabla_z\psi)^+ = \begin{pmatrix} \mathbb{O} & \mathbb{I} \\ -\mathbb{I} & \mathbb{O}  \end{pmatrix} \begin{pmatrix} \mathbb{I} & \nabla_pf \\ \mathbb{O} & \mathbb{I} \end{pmatrix}^T \begin{pmatrix} \mathbb{O} & -\mathbb{I} \\ \mathbb{I} & \mathbb{O}  \end{pmatrix} = \begin{pmatrix} \mathbb{I} & -\nabla_pf \\ \mathbb{O} & \mathbb{I} \end{pmatrix},
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""for the ``G``-SympNet layers, where we assumed that ``\psi`` only changes the ``q`` component. Because these matrices are square, the inverse ``(\nabla\psi)^+ = (\nabla\psi)^{-1}`` is unique.""")
```

The SympNet layers in the symplectic autoencoder operate in *intermediate dimensions* (as well as the input and output dimensions). In the following we explain how `GeometricMachineLearning` computes those intermediate dimensions. 

## Intermediate Dimensions

For a high-fidelity system of dimension ``2N`` and a reduced system of dimension ``2n``, the intermediate dimensions in the symplectic encoder and the decoder are computed according to: 

```julia
iterations = Vector{Int}(n : (N - n) ÷ (n_blocks - 1) : N)
iterations[end] = full_dim2
iterations * 2
```

So for e.g. ``2N = 100,`` ``2n = 10`` and ``\mathtt{n\_blocks} = 3`` we get 

```math
\mathrm{iterations} = 5\mathtt{:}(45 \div 2)\mathtt{:}50 = 5\mathtt{:}22\mathtt{:}50 = (5, 27, 49).
```

We still have to perform the two other modifications in the algorithm above:
1. `iterations[end] = full_dim2` ``\ldots`` assign `full_dim2` to the last entry,
2. `iterations * 2` ``\ldots`` multiply all the intermediate dimensions by two.

The resulting dimensions are:

```math
(10, 54, 100).
```

The second step (the multiplication by two) is needed to arrive at intermediate dimensions that are even. This is necessary to preserve the [canonical symplectic structure of the system](@ref "Symplectic Systems").


## Example

A visualization of an instance of [`SymplecticAutoencoder`](@ref) is shown below: 

```@example 
Main.include_graphics("../tikz/symplectic_autoencoder_architecture"; width = .6, caption = raw"Example of a symplectic autoencoder. The SympNet layers are in green, the PSD-like layers are in blue. ") # hide
```

In this figure we have the following configuration: `n_encoder_blocks` is two, `n_encoder_layers` is four, `n_decoder_blocks` is three and `n_decoder_layers` is two. For a full dimension of 100 and a reduced dimension of ten we can build such an instance of a symplectic autoencoder by calling:

```@example sae
using GeometricMachineLearning

const full_dim = 100
const reduced_dim = 10

model = SymplecticAutoencoder(full_dim, reduced_dim; 
                                                    n_encoder_blocks = 2, 
                                                    n_encoder_layers = 4, 
                                                    n_decoder_blocks = 3, 
                                                    n_decoder_layers = 2)
@assert Chain(model).layers[1] == GradientLayerQ{100, 100, typeof(tanh)}(500, tanh) # hide
@assert Chain(model).layers[2] == GradientLayerP{100, 100, typeof(tanh)}(500, tanh) # hide
@assert Chain(model).layers[3] == GradientLayerQ{100, 100, typeof(tanh)}(500, tanh) # hide
@assert Chain(model).layers[4] == GradientLayerP{100, 100, typeof(tanh)}(500, tanh) # hide
@assert Chain(model).layers[5] == PSDLayer{100, 10}() # hide
@assert Chain(model).layers[6] == GradientLayerQ{10, 10, typeof(tanh)}(50, tanh) # hide
@assert Chain(model).layers[7] == GradientLayerP{10, 10, typeof(tanh)}(50, tanh) # hide
@assert Chain(model).layers[8] == PSDLayer{10, 54}() # hide
@assert Chain(model).layers[9] == GradientLayerQ{54, 54, typeof(tanh)}(270, tanh) # hide
@assert Chain(model).layers[10] == GradientLayerP{54, 54, typeof(tanh)}(270, tanh) # hide
@assert Chain(model).layers[11] == PSDLayer{54, 100}() # hide

for layer in Chain(model)
    println(stdout, layer)
end
```

We also see that the intermediate dimension in the decoder is `54`  for the specified dimensions and `n_decoder_blocks = 3` as was outlined before.

## Library Functions

```@docs
SymplecticAutoencoder
```

## References

```@bibliography
Pages = []
Canonical = false

brantner2023symplectic
```