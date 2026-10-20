# The forward pass of `MultiHeadAttention` allocates what the same head products allocate when
# their keys are constants: it builds no head key per pass. `reference_output` below is the forward
# pass with the key of head `i` read from a tuple that is built once, outside the measurement.
#
# One rebuilt key costs 192 B a pass (Julia 1.13.2, one head), so k = 1000 passes put a key built
# per pass 192 000 B above the reference, 187 times the tolerance of 1024 B. The tolerance absorbs
# the 16-byte quantisation of the figures on some platforms.

using GeometricMachineLearning
using GeometricMachineLearning: params, _mul, tensor_transpose_tensor_mul, tensor_tensor_mul
using Test
import Random

include("../helpers/eltypes.jl")

const PASSES = 1000
const TOLERANCE = 1024

function reference_output(d::MultiHeadAttention, x::AbstractMatrix{T}, ps, head_keys) where {T}
    dim = size(x, 1)
    head_outputs = map(1:(d.n_heads)) do i
        key = head_keys[i]
        _mul(ps.PV[key]', x) *
        d.activation(_mul(ps.PQ[key]', x)' * _mul(ps.PK[key]', x) / T(sqrt(dim)))
    end
    x + reduce(vcat, head_outputs)
end

function reference_output(d::MultiHeadAttention, x::AbstractArray{T, 3}, ps, head_keys) where {T}
    dim = size(x, 1)
    head_outputs = map(1:(d.n_heads)) do i
        key = head_keys[i]
        Q_tensor = _mul(ps.PQ[key]', x)
        K_tensor = _mul(ps.PK[key]', x)
        V_tensor = _mul(ps.PV[key]', x)
        QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)
        tensor_tensor_mul(V_tensor, d.activation(QK_tensor / T(sqrt(dim))))
    end
    x + vcat(head_outputs...)::eltype(head_outputs)
end

# The bytes that `k` calls of `f(d, x, ps)` allocate, after one call that compiles it.
function allocations(f::F, d, x, ps, k::Integer) where {F}
    f(d, x, ps)
    @allocated for _ in 1:k
        f(d, x, ps)
    end
end

forward(d, x, ps) = d(x, ps)

struct WithKeys{K}
    head_keys::K
end
(r::WithKeys)(d, x, ps) = reference_output(d, x, ps, r.head_keys)

@testset "MultiHeadAttention builds no key per pass, n_heads = $n_heads, $(ndims(x₀))-array ($T)" for T in REAL_ELTYPES,
    n_heads in (1, 2),
    x₀ in (zeros(4, 3), zeros(4, 3, 2))

    Random.seed!(n_heads)
    d = MultiHeadAttention(4, n_heads)
    ps = params(NeuralNetwork(Chain(d), CPU(), T))[1]
    x = T.(x₀ .+ reshape(1:length(x₀), size(x₀)) ./ length(x₀))
    reference = WithKeys(ntuple(i -> Symbol("head_", i), n_heads))

    @test reference(d, x, ps) == forward(d, x, ps)
    @test eltype(forward(d, x, ps)) === T

    literal = allocations(reference, d, x, ps, PASSES)
    measured = allocations(forward, d, x, ps, PASSES)
    @test literal > 0
    @test abs(measured - literal) ≤ TOLERANCE
end
