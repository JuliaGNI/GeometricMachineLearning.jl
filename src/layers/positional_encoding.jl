@doc raw"""
    positional_encoding(T, dim, seq_length)

The sinusoidal positional encoding of [vaswani2017attention](@cite), as a `dim × seq_length`
matrix:

```math
P_{2j+1,\,i} = \sin\left(\frac{i - 1}{10000^{2j/d}}\right), \qquad
P_{2j+2,\,i} = \cos\left(\frac{i - 1}{10000^{2j/d}}\right),
```

with ``d`` the feature dimension `dim`. Row pairs share a frequency, and the frequencies decay
geometrically, so a pair of rows encodes the position at one wavelength. The shortest wavelength is
``2\pi``; the longest approaches the ``10000\cdot2\pi`` of the paper only as ``d`` grows, and at
``d = 4`` it is ``100\cdot2\pi``.

Positions are counted from zero — the paper writes only "`pos` is the position" and gives no base,
and zero is the standard reading — so the first column is
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
    positional_encoding(CPU(), T, dim, seq_length)
end

# One entry per index pair, on `backend`: `j = (i - 1) ÷ 2` is the row pair's index.
@kernel function positional_encoding_kernel!(P::AbstractMatrix{T}) where {T}
    i, pos = @index(Global, NTuple)
    # Row pairs `(1,2)`, `(3,4)`, … share the frequency `λ^j`, so the pair index is the row index
    # shifted before the halving. `i ÷ 2` would pair `(2,3)`, `(4,5)`, … instead, and leave row 1
    # unpaired.
    j = (i - 1) ÷ 2
    λʲ = T(10)^(T(-8) * T(j) / T(size(P, 1)))
    θ = T(pos - 1) * λʲ
    P[i, pos] = isodd(i) ? sin(θ) : cos(θ)

    nothing
end

# The encoding as an array of `backend`'s own, filled by the kernel above. `CPU()` gives the
# `Matrix` the three-argument method returns.
function positional_encoding(backend::Backend, ::Type{T}, dim::Integer,
        seq_length::Integer) where {T}
    P = KernelAbstractions.allocate(backend, T, dim, seq_length)
    positional_encoding_kernel!(backend)(P; ndrange = size(P))
    P
end

# The matrix is a constant: its arguments are a type, two lengths and a backend, and none of them is
# a differentiable quantity. Without this declaration Zygote traces into the kernel launch and
# refuses it, so a network carrying a `PositionalEncoding` cannot be trained at all.
ChainRulesCore.@non_differentiable positional_encoding(::Any...)

@doc raw"""
    PositionalEncoding(dim)

A layer that adds the sinusoidal positional encoding of [vaswani2017attention](@cite) to its input.

It has no parameters. The sequence length is read from the input's second axis at every call rather
than fixed at construction, because a transformer here takes the sequence length from its data and
not from its architecture — the same network is applied to trajectories of different lengths. The
encoding matrix is therefore built per call, which costs one `dim × seq_length` allocation on the
input's own backend, and a second for the sum.

Its Jacobian is the identity, since it adds a constant, so it composes with the structure-preserving
architectures without changing what they preserve: a [`LinearSymplecticTransformer`](@ref) with this
layer in front is as symplectic as one without it.

See [`positional_encoding`](@ref) for the matrix itself, and [`Transformer`](@ref), whose
`positional_encoding` keyword puts this layer at the front of the chain.

# Examples

```jldoctest
using GeometricMachineLearning

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

# A matrix and a batch of matrices, and deliberately not a vector. A vector has no second axis to
# read a sequence length from, and broadcasting one against the `M × 1` encoding would return a
# matrix — so the keyword would change the rank of the output and let an input through that
# `MultiHeadAttention` rejects anyway.
function (::PositionalEncoding{M, M})(x::Union{AbstractMatrix, AbstractArray{<:Any, 3}},
        ::NamedTuple) where {M}
    x .+ _encoding_for(x, Val(M))
end

# The encoding matrix of a layer of dimension `M` for `x`, allocated on the backend of `x`. The
# backend comes from `KernelAbstractions.get_backend` and not from `networkbackend`, which answers
# `CPUStatic()` for a static array, and no `positional_encoding` method takes that backend: a static
# array's encoding is a plain `Matrix`, which the broadcast adds into its own type. It is a constant
# of the layer and not a function of the data's values, so it is declared non-differentiable:
# Zygote traces no further, which is what keeps the backend lookup out of the trace as well.
function _encoding_for(x::AbstractArray, ::Val{M}) where {M}
    positional_encoding(KernelAbstractions.get_backend(x), eltype(x), M, size(x, 2))
end

ChainRulesCore.@non_differentiable _encoding_for(::Any, ::Any)

# A `(q, p)` input is stacked, encoded at the full dimension `M`, and split again — which is what
# `MultiHeadAttention` and `ResNetLayer` do with the same input, so a chain accepts `(q, p)` with
# this layer in front exactly as it does without it. Without this method the `positional_encoding`
# keyword would *remove* an input type the transformer otherwise takes.
function (d::PositionalEncoding{M, M})(z::QPT, ps::NamedTuple) where {M}
    @assert size(z.q, 1) * 2 == M
    assign_q_and_p(d(vcat(z.q, z.p), ps), M ÷ 2)
end
