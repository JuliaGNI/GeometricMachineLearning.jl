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
    P = Matrix{T}(undef, dim, seq_length)
    for i in 1:dim
        # Row pairs `(1,2)`, `(3,4)`, … share the frequency `λ^j`, so the pair index is the row
        # index shifted before the halving. `i ÷ 2` would pair `(2,3)`, `(4,5)`, … instead, and
        # leave row 1 unpaired.
        j = (i - 1) ÷ 2
        λʲ = T(10)^(T(-8) * T(j) / T(dim))
        for pos in 1:seq_length
            θ = T(pos - 1) * λʲ
            P[i, pos] = isodd(i) ? sin(θ) : cos(θ)
        end
    end
    P
end

# The matrix is a constant: its arguments are a type and two lengths, and none of them is a
# differentiable quantity. Without this declaration Zygote traces into the loop above and refuses
# the `setindex!`, so a network carrying a `PositionalEncoding` cannot be trained at all.
ChainRulesCore.@non_differentiable positional_encoding(::Any, ::Any, ::Any)

@doc raw"""
    PositionalEncoding(dim)

A layer that adds the sinusoidal positional encoding of [vaswani2017attention](@cite) to its input.

It has no parameters. The sequence length is read from the input's second axis at every call rather
than fixed at construction, because a transformer here takes the sequence length from its data and
not from its architecture — the same network is applied to trajectories of different lengths. The
encoding matrix is therefore built per call, which costs one `dim × seq_length` allocation.

!!! warning "CPU only"
    [`positional_encoding`](@ref) builds a `Matrix`, so this layer adds a host array to whatever it
    is given. Every other layer here allocates through the backend — `KernelAbstractions.allocate`,
    or `similar(x, …)` in the forward pass — and this one does not yet. On a GPU array this is not a
    slowdown but a failure: the broadcast fails to compile, because the host array cannot be read
    from a kernel. A network carrying this layer is therefore a CPU network. Nothing in the test
    suite would catch that, because the suite has no GPU test.

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
    x .+ positional_encoding(eltype(x), M, size(x, 2))
end

# A `(q, p)` input is stacked, encoded at the full dimension `M`, and split again — which is what
# `MultiHeadAttention` and `ResNetLayer` do with the same input, so a chain accepts `(q, p)` with
# this layer in front exactly as it does without it. Without this method the `positional_encoding`
# keyword would *remove* an input type the transformer otherwise takes.
function (d::PositionalEncoding{M, M})(z::QPT, ps::NamedTuple) where {M}
    @assert size(z.q, 1) * 2 == M
    assign_q_and_p(d(vcat(z.q, z.p), ps), M ÷ 2)
end
