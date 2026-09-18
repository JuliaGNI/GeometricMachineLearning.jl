# Positional Encoding

Attention is *permutation-equivariant*: permute the columns of its input and its output is permuted
the same way. On its own it therefore cannot tell one ordering of a sequence from another. The
original transformer paper [vaswani2017attention](@cite) resolves this by adding a fixed matrix to
the input, whose columns encode the position:

```math
P_{2j+1,\,i} = \sin\left(\frac{i - 1}{10000^{2j/d}}\right), \qquad
P_{2j+2,\,i} = \cos\left(\frac{i - 1}{10000^{2j/d}}\right),
```

with ``d`` the feature dimension. Each pair of rows encodes the position at one wavelength, and the
wavelengths grow geometrically from ``2\pi`` to ``10000\cdot2\pi``, so distant positions are
distinguished by the slow rows and neighbouring ones by the fast rows.

## Why it is off by default

The transformers in this package are usually applied to **phase-space trajectories**, where the
ordering is carried by the data rather than needing to be encoded: a
[`TransformerIntegrator`](@ref) is handed a time series whose columns are consecutive time steps.
So [`Transformer`](@ref) takes `positional_encoding = false` unless asked otherwise.

## It does not disturb the geometry

The layer adds a constant, so its Jacobian is the identity. A structure-preserving architecture with
this layer in front therefore preserves exactly what it preserved without it — a
[`LinearSymplecticTransformer`](@ref) stays symplectic, a [`VolumePreservingTransformer`](@ref)
stays volume-preserving. This is asserted in the test suite rather than only argued here.

The layer holds no parameters, and the sequence length is read from the input at every call rather
than fixed when the layer is built, because a network here is applied to trajectories of whatever
length the data has.

```@docs
PositionalEncoding
positional_encoding
```

```@raw latex
\begin{comment}
```

## References

```@bibliography
Pages = []
Canonical = false

vaswani2017attention
```

```@raw latex
\end{comment}
```
