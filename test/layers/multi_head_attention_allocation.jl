# The forward pass of `MultiHeadAttention` allocates what its head products allocate and nothing
# more: the same products, written below with the head keys as constants, allocate the same bytes.
# A forward pass that builds each key on each pass, `Symbol("head_", i)`, and collects the heads in
# a `Vector` allocates 256 000 B more over k = 1000 passes with one head, and 656 000 B more with
# two, on Julia 1.13.2: 250 and 640 times the tolerance of 1024 B, which absorbs the quantisation
# of `@allocated` and nothing of that size.

using GeometricMachineLearning
using GeometricMachineLearning: _mul, tensor_transpose_tensor_mul, tensor_tensor_mul, params
using Test
import Random

include("../helpers/eltypes.jl")

const PASSES = 1000
const TOLERANCE = 1024

function head(d, x::AbstractMatrix{T}, PQ, PK, PV) where {T}
    _mul(PV', x) * d.activation(_mul(PQ', x)' * _mul(PK', x) / T(sqrt(size(x, 1))))
end

function head(d, x::AbstractArray{T, 3}, PQ, PK, PV) where {T}
    QK = tensor_transpose_tensor_mul(_mul(PQ', x), _mul(PK', x))
    tensor_tensor_mul(_mul(PV', x), d.activation(QK / T(sqrt(size(x, 1)))))
end

function constant_keys(d, x, ps, ::Val{1})
    x + vcat(head(d, x, ps.PQ.head_1, ps.PK.head_1, ps.PV.head_1))
end

function constant_keys(d, x, ps, ::Val{2})
    x + vcat(head(d, x, ps.PQ.head_1, ps.PK.head_1, ps.PV.head_1),
        head(d, x, ps.PQ.head_2, ps.PK.head_2, ps.PV.head_2))
end

forward(d, x, ps, ::Val) = d(x, ps)

function passes(f::F, d, x, ps, v, k) where {F}
    for _ in 1:k
        f(d, x, ps, v)
    end
end

function allocations(f::F, d, x, ps, v) where {F}
    (passes(f, d, x, ps, v, 1); @allocated passes(
        f, d, x, ps, v, PASSES))
end

@testset "MultiHeadAttention allocates what its head products allocate ($T, $n_heads heads, $(ndims(x)) axes)" for T in REAL_ELTYPES,
    n_heads in (1, 2),
    x in (rand(Random.Xoshiro(0), T, 4, 3), rand(Random.Xoshiro(0), T, 4, 3, 2))

    d = MultiHeadAttention(4, n_heads)
    ps = params(NeuralNetwork(Chain(d), CPU(), T)).L1
    v = Val(n_heads)

    @test forward(d, x, ps, v) == constant_keys(d, x, ps, v)

    literal = allocations(constant_keys, d, x, ps, v)
    measured = allocations(forward, d, x, ps, v)
    @test literal > 0
    @test abs(measured - literal) ≤ TOLERANCE
end
