# Volume-Preserving Transformer

The volume-preserving transformer [brantner2024volume](@cite) is, similar to the standard transformer, a combination of two different neural networks: a [volume-preserving attention layer](@ref "Volume-Preserving Attention") and a [volume-preserving feedforward layer](@ref "Volume-Preserving Feedforward Neural Network"). It is visualized below:

```@example 
Main.include_graphics("../tikz/vp_transformer"; caption = raw"Visualization of the Volume-Preserving Transformer architecture", width = .25) # hide
```

In the figure we indicate that we leave out the *add connection*. When talking about the [`standard transformer`](@ref "Standard Transformer") we said that the add connection is optional and can be included via the keyword argument `add_connection`. For the volume-preserving transformer this is not true: it is always excluded. 

Note that the volume-preserving transformer preserves the volume in the sense of the product spaces. That the [`VolumePreservingAttention`](@ref) layer preserves this structure was discussed when we [introduced it](@ref "Volume-Preserving Attention"). That the [`VolumePreservingFeedForwardLayer`](@ref)s preserve this structure on the product space is also easy to see. We take a [`VolumePreservingFeedForwardLayer`](@ref), e.g. 
```math
    \psi: z \mapsto \sigma(Lz + b),
```
and look at its action on the element of the product space ``[z^{(1)}, \ldots, z^{(T)}] = Z\in\mathbb{R}^{d\times{}T}``:
```math
    \psi_T: [z^{(1)}, \ldots, z^{(T)}] \mapsto [\psi(z^{(1)}), \ldots, \psi(z^{(T)})].
```

The jacobian of ``\hat{\psi}_T:\mathbb{R}^{dT}\to\mathbb{R}^{dT}``, the representation of ``\psi_T`` in the *coordinate system* of the [big vector](@ref "How is Structure Preserved?"), is of the form[^1]

[^1]: In order to arrive at this representation we possibly have to exchange the order of the rows in the matrix. This is however not critical since it may only cause a sign change in the determinant.

```math
    \nabla\hat{\psi}_T = \begin{bmatrix} J & \mathbb{O} & \cdots & \mathbb{O} \\ 
                                         \mathbb{O} & J & \cdots & \mathbb{O} \\ 
                                         \vdots & \ddots & \vdots & \vdots \\
                                         \mathbb{O} & \mathbb{O} & \cdots & J\end{bmatrix} = J \otimes \mathbb{I}_T,
```
where ``J`` is the jacobian of ``\psi``. We now see that ``\mathrm{det}(\nabla\hat{\psi}_T) = 1`` and volume in the product space is preserved. 

## Library Functions 

```@docs
VolumePreservingTransformer
```

## References 

```@bibliography
Pages = []
Canonical = false

brantner2024volume
```