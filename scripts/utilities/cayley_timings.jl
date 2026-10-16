# The forward and the backward time of the Cayley transform of a batch of 1000 skew slices at scale
# 1, `tensor_cayley`, against the closed form the tree defines for that slice size: `tensor_cayleyN`
# for N = 2 to 5. A size whose closed form the tree does not define, N = 6 to 8, prints the kernel's
# own time and no ratio. BLAS is pinned to one thread, and every variant, every N and every element
# type runs in a fresh process; a process makes one warm-up call it does not time, then takes the
# median of five timed calls. The parent prints the ratio of the kernel to the closed form, forward
# and backward.
#
# The table that decided which closed forms stay carried a column for N = 6 to 8 as well, against the
# batch inverse of a long slice, which the tree no longer defines. That column came from a commit
# that still held it, and the `CHANGELOG.md` entry of the change names the commit.
#
# Run from the repository root, with the package's own environment, which carries `Zygote` and the
# standard libraries this script uses; `--project=scripts` resolves them as well, and CUDA with them:
#   julia --startup-file=no --project=. scripts/utilities/cayley_timings.jl
#   julia --startup-file=no --project=. scripts/utilities/cayley_timings.jl tensor_cayley 4 Float32
#
# Metal has no reference for N = 6 to 8: the batch inverse that served those rows called `inv` inside
# a kernel, and the tree does not define it, so those rows print the kernel's time alone. N = 2 to 5
# compare against the closed forms, which run on Metal as well; `Metal` is not a dependency of the
# package, so its column runs `measure` through Kaimon, in a scratch environment that develops this
# worktree and adds Metal. This file runs its table only when it is not included, so a session
# reaches `measure` by including it and calling it:
#
#   julia> include("scripts/utilities/cayley_timings.jl")
#   julia> measure("tensor_cayley", 6, Float32, Metal.@sync)
#
# `Metal.@sync` is what makes the figure a time rather than a launch: without it the call returns
# before the kernel has run.

using GeometricMachineLearning
using LinearAlgebra: BLAS
using Printf
import Random
import Zygote

const KERNEL = "tensor_cayley"
const SLICES = 1000
const RUNS = 5

median(v) = sort(v)[(length(v) + 1) ÷ 2]

# One untimed warm-up call, then `RUNS` timed calls, in µs. `sync` wraps a timed call and is the
# identity except on Metal.
function times(f, sync = identity)
    f()
    median([(@elapsed sync(f())) * 1e6 for _ in 1:RUNS])
end

function skew_batch(::Type{T}, n) where {T}
    R = randn(T, n, n, SLICES)
    R - permutedims(R, (2, 1, 3))
end

# The forward and the backward time of one variant in this process, in µs.
function measure(variant, n, ::Type{T}, sync = identity) where {T}
    BLAS.set_num_threads(1)
    Random.seed!(0)

    C = skew_batch(T, n)
    f = getfield(GeometricMachineLearning, Symbol(variant))

    forward = times(() -> f(C), sync)
    Λ, pullback = Zygote.pullback(f, C)
    Λ̄ = rand(T, size(Λ))
    backward = times(() -> pullback(Λ̄), sync)

    forward, backward
end

# The closed form of the transform of an `n × n` slice, where the tree defines one.
reference(n) = n <= 5 ? "tensor_cayley$n" : nothing

function one_process(variant, n, T)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) $variant $n $T`
    parse.(Float64, split(readchomp(cmd)))
end

function table()
    @printf("%-17s %-2s %-8s %11s %11s  %s\n",
        "variant", "N", "T", "forward µs", "backward µs",
        "ratio kernel/closed form")
    for T in (Float32, Float64), n in 2:8

        kernel = one_process(KERNEL, n, T)
        name = reference(n)
        if name !== nothing && isdefined(GeometricMachineLearning, Symbol(name))
            closed = one_process(name, n, T)
            @printf("%-17s %-2d %-8s %11.1f %11.1f  forward %.2f, backward %.2f\n",
                KERNEL, n, T, kernel[1], kernel[2], kernel[1] / closed[1],
                kernel[2] / closed[2])
        else
            @printf("%-17s %-2d %-8s %11.1f %11.1f  no closed form\n",
                KERNEL, n, T, kernel[1], kernel[2])
        end
    end
end

if length(ARGS) == 3
    println(join(measure(ARGS[1], parse(Int, ARGS[2]), getfield(Base, Symbol(ARGS[3]))), " "))
elseif !isinteractive()
    table()
end
