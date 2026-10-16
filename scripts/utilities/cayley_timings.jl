# The forward and the backward time of the Cayley transform of a batch of 1000 skew slices at scale
# 1, `tensor_cayley`, against the closed forms it would replace: `tensor_cayleyN` for N = 2 to 5 and
# `cpu_tensor_cayley` for N = 6 to 8, each where the tree still defines it. BLAS is pinned to one
# thread, and every variant, every N and every element type runs in a fresh process; a process makes
# one warm-up call it does not time, then takes the median of five timed calls. The parent prints the
# ratio of the kernel to the closed form, forward and backward.
#
# The closed forms that the kernel replaces are deleted by the outcome this table decides, so the
# figures come from a branch commit that holds both the kernel and every closed form, and the
# `CHANGELOG.md` entry of that outcome cites the hash of that commit. After the deletions the script
# times only the variants still defined (`isdefined`).
#
# Run from the repository root, with the scripts environment; one process of one variant is the
# second form, which the first starts:
#   julia --startup-file=no --project=scripts scripts/utilities/cayley_timings.jl
#   julia --startup-file=no --project=scripts scripts/utilities/cayley_timings.jl tensor_cayley 4 Float32
#
# Metal has no reference for N = 6 to 8, because `cpu_tensor_cayley` calls `inv` inside a kernel, and
# Metal is not a dependency of `scripts/Project.toml`. The Metal column runs `measure` through
# Kaimon, in a scratch environment that develops the worktree and adds Metal, as
#
#   julia> measure("tensor_cayley", 6, Float32, Metal.@sync)
#
# because a Metal launch returns before the kernel has run. That column prints the kernel's own time
# and no ratio.

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

# The closed form of the transform of an `n × n` slice, where the tree still defines it.
function reference(n)
    n <= 5 ? "tensor_cayley$n" : "cpu_tensor_cayley"
end

function one_process(variant, n, T)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) $variant $n $T`
    parse.(Float64, split(readchomp(cmd)))
end

if length(ARGS) == 3
    println(join(measure(ARGS[1], parse(Int, ARGS[2]), getfield(Base, Symbol(ARGS[3]))), " "))
else
    @printf("%-17s %-2s %-8s %11s %11s  %s\n",
        "variant", "N", "T", "forward µs", "backward µs",
        "ratio kernel/closed form")
    for T in (Float32, Float64), n in 2:8

        kernel = one_process(KERNEL, n, T)
        name = reference(n)
        if isdefined(GeometricMachineLearning, Symbol(name))
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
