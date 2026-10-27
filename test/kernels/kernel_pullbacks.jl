using GeometricMachineLearning: tensor_tensor_mul, tensor_transpose_tensor_mul,
                                tensor_mat_skew_sym_assign, tensor_cayley, _mul
using ChainRulesCore: rrule, AbstractZero, NoTangent, ZeroTangent, @thunk
using ChainRulesTestUtils
using FiniteDifferences: central_fdm, j′vp
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

const verbose = false

# `skew_batch`, a skew batch for the Cayley rule, whose slices are square.
include("../helpers/cayley.jl")

function main(first_dim, second_dim, third_dim, third_tensor_dim)
    test_rrule(tensor_tensor_mul, rand(first_dim, second_dim, third_tensor_dim),
        rand(second_dim, third_dim, third_tensor_dim))
    test_rrule(tensor_transpose_tensor_mul, rand(second_dim, first_dim, third_tensor_dim),
        rand(second_dim, third_dim, third_tensor_dim))
    test_rrule(
        tensor_mat_skew_sym_assign, rand(first_dim, second_dim, third_tensor_dim + 1),
        rand(first_dim, first_dim))
    test_rrule(_mul, rand(first_dim, second_dim)', rand(first_dim, third_dim, third_tensor_dim))
    test_rrule(_mul, rand(first_dim, second_dim)', rand(first_dim, third_dim))
    test_rrule(tensor_cayley, skew_batch(Float64, first_dim, third_tensor_dim))
end

const dim_range = 10
const num_tests = 10
for _ in 1:num_tests
    first_dim = Int(ceil(dim_range * rand()))
    second_dim = Int(ceil(dim_range * rand()))
    third_dim = Int(ceil(dim_range * rand()))
    third_tensor_dim = Int(ceil(dim_range * rand()))
    verbose && println("dims are : (", first_dim, ", ", second_dim,
        ", ", third_dim, ", ", third_tensor_dim, ")")
    main(first_dim, second_dim, third_dim, third_tensor_dim)
    verbose && println()
end

# Every `rrule` of `src/kernels/` with a primal of element type `T`, and its arguments.
function rules(T)
    ((tensor_tensor_mul, (rand(T, 3, 4, 2), rand(T, 4, 5, 2))),
        (tensor_transpose_tensor_mul, (rand(T, 4, 3, 2), rand(T, 4, 5, 2))),
        (tensor_mat_skew_sym_assign, (rand(T, 3, 4, 2), rand(T, 3, 3))),
        (_mul, (rand(T, 4, 3)', rand(T, 4, 5, 2))),
        (tensor_cayley, (skew_batch(T, 3, 2),)))
end

@testset "a zero seed gives a zero cotangent for every argument" begin
    for T in REAL_ELTYPES, (f, args) in rules(T),
        seed in (ZeroTangent(), NoTangent(),
            @thunk(ZeroTangent()))
        c̄ = rrule(f, args...)[2](seed)
        @test c̄ isa Tuple
        @test length(c̄) == 1 + length(args)
        @test all(c -> c isa AbstractZero, c̄)
    end
end

@testset "a seed of the other precision gives cotangents of the primal's element type" begin
    for (T, S) in ((Float32, Float64), (Float64, Float32)), (f, args) in rules(T)

        y, pullback = rrule(f, args...)
        c̄ = pullback(S.(y))
        @test c̄[1] isa NoTangent
        for c in Base.tail(c̄)
            @test eltype(c) === T
        end
    end
end

# The seed is converted before the pullback computes, so a `Float64` seed gives exactly the
# cotangents of its `Float32` rounding. The last row cancels: computed in `Float64`, the cotangent
# of `W` keeps a residue of `eps(Float32) / 4` that the rounded seed does not have.
@testset "a seed of the other precision is converted at entry" begin
    for (f, args) in rules(Float32)
        y, pullback = rrule(f, args...)
        ȳ = Float64.(rand(Float32, size(y))) .+ eps(Float32) / 4
        @test pullback(ȳ) == pullback(Float32.(ȳ))
    end
    W = ones(Float32, 2, 1)
    x = Float32[1 -1; 0 0]
    ȳ = [1 + Float64(eps(Float32)) / 4 1.0]
    pullback = rrule(_mul, W', x)[2]
    @test ȳ isa Matrix{Float64}
    @test pullback(ȳ) == pullback(Float32.(ȳ))
end

# The `Float64` rules are checked by `test_rrule` above; the `Float32` ones against finite
# differences in `Float64` on the same data, at the accuracy of `Float32`.
@testset "Float32 cotangents against Float64 finite differences" begin
    for (f, args) in rules(Float32)
        y, pullback = rrule(f, args...)
        ȳ = rand(Float32, size(y))
        c̄ = Base.tail(pullback(ȳ))
        args64 = map(a -> Float64.(a), args)
        fd = j′vp(central_fdm(5, 1), f, Float64.(ȳ), args64...)
        for (c, reference) in zip(c̄, fd)
            if c isa AbstractZero
                @test maximum(abs, reference) < sqrt(eps(Float32))
            else
                @test eltype(c) === Float32
                @test isapprox(c, reference; rtol = sqrt(eps(Float32)))
            end
        end
    end
end
