@doc raw"""
    positional_encoding(T, dim, seq_length)

The sinusoidal positional encoding of [vaswani2017attention](@cite), as a `dim × seq_length`
matrix:

```math
P_{2j+1,\,i} = \sin\left(\frac{i - 1}{10000^{2j/d}}\right), \qquad
P_{2j+2,\,i} = \cos\left(\frac{i - 1}{10000^{2j/d}}\right),
```

with ``d`` the feature dimension `dim`. Row pairs share a frequency, and the frequencies decay
geometrically, so a pair of rows encodes the position at one wavelength and the matrix as a whole
spans wavelengths from ``2\pi`` to ``10000\cdot2\pi``.

Positions are counted from zero, as in the paper, so the first column is
``(0, 1, 0, 1, \ldots)``.

# Examples

```jldoctest
using GeometricMachineLearning

positional_encoding(Float32, 4, 3)[:, 1]

# output

4-element Vector{Float32}:
 0.0
 1.0
 0.0
 1.0
```
"""
function positional_encoding(::Type{T}, dim::Integer, seq_length::Integer) where {T}
    P = Matrix{T}(undef, dim, seq_length)
    for i in 1:dim
        # Row pairs `(1,2)`, `(3,4)`, … share the frequency `λ^j`. A version of this in the
        # repository's `legacy/` tree paired them off by one — its first row was an unpaired cosine
        # — which is why this is written from the paper rather than carried over.
        j = (i - 1) ÷ 2
        λʲ = T(10)^(T(-8) * T(j) / T(dim))
        for pos in 1:seq_length
            θ = T(pos - 1) * λʲ
            P[i, pos] = isodd(i) ? sin(θ) : cos(θ)
        end
    end
    P
end

@doc raw"""
    PositionalEncoding(dim)

A layer that adds the sinusoidal positional encoding of [vaswani2017attention](@cite) to its input.

It has no parameters. The sequence length is read from the input's second axis at every call rather
than fixed at construction, because a transformer here takes the sequence length from its data and
not from its architecture — the same network is applied to trajectories of different lengths. The
encoding matrix is therefore built per call, which costs one `dim × seq_length` allocation.

Its Jacobian is the identity, since it adds a constant, so it composes with the structure-preserving
architectures without changing what they preserve: a [`LinearSymplecticTransformer`](@ref) with this
layer in front is as symplectic as one without it.

See [`positional_encoding`](@ref) for the matrix itself, and [`Transformer`](@ref), whose
`positional_encoding` keyword puts this layer at the front of the chain.

# Examples

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: initialparameters
import Random

l = PositionalEncoding(4)
x = zeros(Float32, 4, 3)

l(x, NamedTuple())[:, 1]

# output

4-element Vector{Float32}:
 0.0
 1.0
 0.0
 1.0
```
"""
struct PositionalEncoding{M, N} <: AbstractExplicitLayer{M, N} end

PositionalEncoding(dim::Integer) = PositionalEncoding{dim, dim}()

function initialparameters(::AbstractRNG, ::AbstractNeuralNetworks.Initializer,
        ::PositionalEncoding, ::Backend, ::Type{T}) where {T}
    NamedTuple()
end

parameterlength(::PositionalEncoding) = 0

function (::PositionalEncoding{M, M})(x::AbstractArray, ::NamedTuple) where {M}
    x .+ positional_encoding(eltype(x), M, size(x, 2))
end
