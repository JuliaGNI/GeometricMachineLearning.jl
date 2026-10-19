# `MultiHeadAttention` builds its head keys once, in `initialparameters`, and its forward pass reads
# them off the parameter `NamedTuple`s. The check below measures that: over 1000 passes a forward
# pass allocates what the same head products with the keys written out allocate, to within 1024
# bytes.
#
# A key built per head per pass is not visible in any value, so the allocation is the only
# observable. One key costs at least 240 bytes a pass, so a rebuilt key exceeds the tolerance
# below by more than two orders of magnitude.

using GeometricMachineLearning
using GeometricMachineLearning: compute_output_of_mha, _mul, tensor_tensor_mul,
                                tensor_transpose_tensor_mul
using AbstractNeuralNetworks: Initializer
using KernelAbstractions
using Test
import Random

GML = GeometricMachineLearning

Random.seed!(1234)

mha_parameters(d, T) = GML.params(NeuralNetwork(Chain(d), CPU(), T))[1]

# One head's output with its three weights given, and the keys written out at the call below: the
# expression the layer computes for a head.
function head_output(d, PQ, PK, PV, x::AbstractMatrix)
    _mul(PV', x) * d.activation(_mul(PQ', x)' * _mul(PK', x) / eltype(x)(sqrt(size(x, 1))))
end

function head_output(d, PQ, PK, PV, x::AbstractArray{T, 3}) where {T}
    Q_tensor = _mul(PQ', x)
    K_tensor = _mul(PK', x)
    V_tensor = _mul(PV', x)
    tensor_tensor_mul(V_tensor,
        d.activation(tensor_transpose_tensor_mul(Q_tensor, K_tensor) / T(sqrt(size(x, 1)))))
end

# The same products, with `ps.PQ.head_1` in place of a key built from the head index.
function constant_keys(d, x, ps, ::Val{1})
    reduce(vcat, (head_output(d, ps.PQ.head_1, ps.PK.head_1, ps.PV.head_1, x),))
end
function constant_keys(d, x, ps, ::Val{2})
    reduce(vcat,
        (head_output(d, ps.PQ.head_1, ps.PK.head_1, ps.PV.head_1, x),
            head_output(d, ps.PQ.head_2, ps.PK.head_2, ps.PV.head_2, x)))
end

# The tensor method concatenates with a variadic `vcat`, so the reference does too.
function constant_keys_tensor(d, x, ps, ::Val{1})
    vcat(head_output(d, ps.PQ.head_1, ps.PK.head_1, ps.PV.head_1, x))
end
function constant_keys_tensor(d, x, ps, ::Val{2})
    vcat(head_output(d, ps.PQ.head_1, ps.PK.head_1, ps.PV.head_1, x),
        head_output(d, ps.PQ.head_2, ps.PK.head_2, ps.PV.head_2, x))
end

# The allocation of `k` calls, measured through a function barrier: the arguments are one concrete
# tuple, so the calls under the timer have concrete types, and the first call warms them.
const PASSES = 1000
function allocations(f::F, args::A, k::Integer) where {F, A}
    f(args...)
    @allocated for _ in 1:k
        f(args...)
    end
end

@testset "a forward pass allocates what literal head keys allocate ($T, n_heads = $n_heads)" for T in (
        Float32, Float64),
    n_heads in (1, 2)

    d = MultiHeadAttention(8, n_heads)
    ps = mha_parameters(d, T)

    for (data, reference) in ((rand(T, 8, 6), constant_keys),
        (rand(T, 8, 6, 3), constant_keys_tensor))
        measured = allocations(compute_output_of_mha, (d, data, ps), PASSES)
        literal = allocations(reference, (d, data, ps, Val(n_heads)), PASSES)

        # The control: the reference allocates a result per pass, so the barrier sees allocations.
        @test literal > 0
        @test abs(measured - literal) <= 1024
    end
end

@testset "one initialparameters builds the head keys once ($T)" for T in (Float32, Float64)
    # One method for `MultiHeadAttention`, whatever the `Stiefel` flag: the flag is a type parameter
    # the method branches on. `AbstractNeuralNetworks` defines a fallback for any `Model`, so the
    # count is over the methods this package defines for the layer.
    matching = filter(m -> m.module === GeometricMachineLearning,
        methods(initialparameters,
            (Random.AbstractRNG, Initializer,
                MultiHeadAttention, KernelAbstractions.Backend,
                Type)))
    @test length(matching) == 1

    for Stiefel in (false, true)
        d = MultiHeadAttention(8, 2; Stiefel)
        ps = mha_parameters(d, T)
        @test keys(ps) == (:PQ, :PK, :PV)
        for branch in (ps.PQ, ps.PK, ps.PV)
            @test keys(branch) == (:head_1, :head_2)
            for W in values(branch)
                if Stiefel
                    @test W isa StiefelManifold
                    @test size(W.A) == (8, 4)
                else
                    @test W isa Matrix{T}
                    @test size(W) == (8, 4)
                end
            end
        end
    end
end

@testset "the forward pass gives what the literal keys give ($T, Stiefel = $Stiefel)" for T in (
        Float32, Float64),
    Stiefel in (false, true)
    # The allocation equality above would also hold between two wrong answers; this pins the answer.
    # Both leaves: a dense weight and a `StiefelManifold` one.
    d = MultiHeadAttention(8, 2; Stiefel)
    ps = mha_parameters(d, T)
    x = rand(T, 8, 5)
    x_tensor = rand(T, 8, 5, 2)
    @test compute_output_of_mha(d, x, ps) == constant_keys(d, x, ps, Val(2))
    @test compute_output_of_mha(d, x_tensor, ps) ==
          constant_keys_tensor(d, x_tensor, ps, Val(2))
end
