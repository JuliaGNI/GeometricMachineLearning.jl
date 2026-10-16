# The forward and the backward time of the products in `src/kernels/mul.jl` at 10×10×1000, BLAS
# on one thread: a left product with each weight kind, a right product with a dense and a symmetric
# weight, and the pullback of `tensor_mat_skew_sym_assign`.
#
# Each variant runs in five fresh processes. A process makes one warm-up call, then takes the
# minimum of 100 timed calls of the forward product and of its backward pass, the
# `Zygote.gradient` through a `NetworkParameters`, with the hook of NeuralNetworkParameters. The
# table gives the median of the five minima in µs. `tensor_mat_skew_sym_assign` has a backward
# column only: its row times the pullback alone.
#
# Run from the repository root, with the scripts environment; one process of one variant is the
# second form, which the first starts:
#   julia --startup-file=no --project=scripts scripts/utilities/product_timings.jl
#   julia --startup-file=no --project=scripts scripts/utilities/product_timings.jl left:SymmetricMatrix Float32

using GeometricMachineLearning
using GeometricMachineLearning: _mul, tensor_mat_skew_sym_assign
using ChainRulesCore: rrule
using LinearAlgebra: BLAS
using NeuralNetworkParameters: NetworkParameters
using Printf
import Random
import Zygote

const VARIANTS = ("left:Matrix", "left:SymmetricMatrix", "left:SkewSymMatrix",
    "left:StrictlyLowerTriangular", "left:StrictlyUpperTriangular", "right:Matrix",
    "right:SymmetricMatrix", "pullback:tensor_mat_skew_sym_assign")
const PROCESSES = 5
const CALLS = 100

minimum_time(f) = (f(); minimum(_ -> @elapsed(f()), 1:CALLS) * 1e6)
median(v) = sort(v)[(length(v) + 1) ÷ 2]

function weight(kind, ::Type{T}) where {T}
    kind == "Matrix" ? rand(T, 10, 10) :
    rand(getfield(GeometricMachineLearning, Symbol(kind)){T}, 10)
end

# The forward and the backward minimum of one variant, in µs, in this process.
function measure(variant, ::Type{T}) where {T}
    BLAS.set_num_threads(1)
    Random.seed!(0)
    side, kind = split(variant, ':')
    x = rand(T, 10, 10, 1000)
    if side == "pullback"
        pullback = rrule(tensor_mat_skew_sym_assign, x, rand(T, 10, 10))[2]
        dB = rand(T, 10, 10, 1000)
        return NaN, minimum_time(() -> pullback(dB))
    end
    product = side == "left" ? (W -> _mul(W, x)) : (W -> _mul(x, W))
    ps = NetworkParameters((L1 = (W = weight(kind, T),),))
    forward = minimum_time(() -> product(ps.L1.W))
    backward = minimum_time(() -> Zygote.gradient(p -> sum(abs2, product(p.L1.W)), ps))
    forward, backward
end

function one_process(variant, T)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) $variant $T`
    parse.(Float64, split(readchomp(cmd)))
end

if length(ARGS) == 2
    println(join(measure(ARGS[1], getfield(Base, Symbol(ARGS[2]))), " "))
else
    @printf("%-38s %-8s %12s %12s\n", "variant", "T", "forward µs", "backward µs")
    for variant in VARIANTS, T in (Float32, Float64)

        runs = [one_process(variant, T) for _ in 1:PROCESSES]
        @printf("%-38s %-8s %12.1f %12.1f\n", variant, T,
            median(first.(runs)), median(last.(runs)))
    end
end
