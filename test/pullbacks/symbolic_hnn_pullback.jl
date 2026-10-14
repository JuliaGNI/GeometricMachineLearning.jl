using GeometricMachineLearning
using GeometricMachineLearning: HNNLoss, ZygotePullback, SigmoidActivation
using AbstractNeuralNetworks: NeuralNetwork, params
using LinearAlgebra: norm
using NeuralNetworkParameters: NetworkParameters
using Test
import Random

include("../helpers/eltypes.jl")

# About 1.8e19 in `Float32` and 1.8e154 in `Float64`: the scale at which the product of the two
# norms overflows and each norm does not.
extreme_scale(::Type{Float32}) = Float32(1e20)
extreme_scale(::Type{Float64}) = Float64(1e155)

# The scale at which the seed scalar `1/(‖R‖‖Y‖)` underflows to zero while a gradient entry does
# not: `Zygote`'s own `norm` rule underflows there too, so the row asserts only that the gradient
# survives.
extreme_scale_underflow(::Type{Float32}) = Float32(1e23)
extreme_scale_underflow(::Type{Float64}) = Float64(1e170)

# The pullback does not depend on the element type, so every testset shares one build.
const arch = StandardHamiltonianArchitecture(2, 4, 1, SigmoidActivation())
const pullback = SymbolicPullback(arch)

# The symbolic pullback is the gradient of `HNNLoss` on the batch: the generated per-sample
# gradients of the surrogate `½‖rᵢ‖²`, scaled by `1/(‖R‖‖Y‖)`. Two batch sizes: at `N = 1` the
# sum is a single per-sample gradient and the scale still divides, while `N = 5` is where the
# gradient of the per-sample sum and the gradient of the batch loss differ.
@testset "SymbolicPullback of a Hamiltonian network is HNNLoss's gradient ($T)" for T in REAL_ELTYPES
    for N in (1, 5)
        Random.seed!(21)
        nn = NeuralNetwork(arch, CPU(), T)
        ps = params(nn)
        X = randn(Random.Xoshiro(11), T, 2, N)
        Y = randn(Random.Xoshiro(12), T, 2, N)

        value, back = pullback(ps, nn.model, (X, Y))
        g = back(one(value))
        zygote_value, zygote_back = ZygotePullback(HNNLoss(arch))(
            ps, nn.model, (X, Y))
        zg = zygote_back(one(zygote_value))

        @test value isa T
        @test value ≈ zygote_value rtol = T(√eps(T))
        @test g isa NetworkParameters
        @test keys(g) == keys(ps)
        for (name, leaf) in pairs(g)
            for k in keys(leaf)
                @test eltype(leaf[k]) === T
                @test leaf[k] ≈ zg[name][k] rtol = T(√eps(T))
            end
        end

        # the value is `HNNLoss` itself, so a wrong scale cannot hide behind a different value
        hvf = GeometricMachineLearning.hamiltonian_vector_field(arch)
        @test value ≈ norm(hvf(X, ps) - Y) / norm(Y) rtol = T(√eps(T))
    end
end

# The two norms are divided by separately, and each of them divides a gradient entry rather than
# the seed: their product overflows where neither norm does, and `inv` of the product is then a zero
# gradient. The scalar `1/(‖R‖‖Y‖)` is zero at the second scale, where the per-entry divisions keep
# every leaf finite and non-zero. Neither scale is a power of two, so round-off shows.
@testset "the scale divides by each norm ($T)" for T in REAL_ELTYPES
    Random.seed!(22)
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    X = randn(Random.Xoshiro(13), T, 2, 5)

    # (the scale, whether the two pullbacks still agree there): `Zygote`'s `norm` rule forms the
    # seed scalar too, so underflowing it takes both paths down together
    rows = ((extreme_scale(T), true), (extreme_scale_underflow(T), false))
    for (scale, agreement) in rows
        Y = randn(Random.Xoshiro(14), T, 2, 5) .* scale

        value, back = pullback(ps, nn.model, (X, Y))
        g = back(one(value))

        @test value isa T
        @test isfinite(value)
        for (name, leaf) in pairs(g)
            for k in keys(leaf)
                @test eltype(leaf[k]) === T
                @test all(isfinite, leaf[k])
                @test any(!iszero, leaf[k])
            end
        end

        agreement || continue
        zygote_value, zygote_back = ZygotePullback(HNNLoss(arch))(ps, nn.model, (X, Y))
        zg = zygote_back(one(zygote_value))
        @test value ≈ zygote_value rtol = T(√eps(T))
        for (name, leaf) in pairs(g)
            for k in keys(leaf)
                @test leaf[k] ≈ zg[name][k] rtol = T(√eps(T))
            end
        end
    end
end

# Data the network's own vector field generates: the residual is zero, the value is zero, and the
# gradient is the zero set rather than `NaN` from an unguarded `inv`.
@testset "a zero residual gives a zero gradient ($T)" for T in REAL_ELTYPES
    Random.seed!(23)
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    X = randn(Random.Xoshiro(17), T, 2, 5)
    Y = GeometricMachineLearning.hamiltonian_vector_field(arch)(X, ps)

    value, back = pullback(ps, nn.model, (X, Y))
    g = back(one(value))

    @test value isa T
    @test value == zero(T)
    @test g isa NetworkParameters
    for (name, leaf) in pairs(g)
        for k in keys(leaf)
            @test eltype(leaf[k]) === T
            @test all(iszero, leaf[k])
        end
    end
end

# The gradient is linear in the seed, as `ZygotePullback`'s is, and a seed of the parameters'
# element type comes back in that element type whatever seed `Δ` is given.
@testset "the seed scales the gradient ($T)" for T in REAL_ELTYPES
    Random.seed!(24)
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    X = randn(Random.Xoshiro(19), T, 2, 5)
    Y = randn(Random.Xoshiro(20), T, 2, 5)

    value, back = pullback(ps, nn.model, (X, Y))
    _, zygote_back = ZygotePullback(HNNLoss(arch))(ps, nn.model, (X, Y))

    @test value isa T
    g₁ = back(one(value))
    g₂ = back(T(2))
    zg₂ = zygote_back(T(2))

    for (name, leaf) in pairs(g₂)
        for k in keys(leaf)
            @test eltype(leaf[k]) === T
            @test eltype(g₁[name][k]) === T
            @test leaf[k] ≈ 2 .* g₁[name][k] rtol = T(√eps(T))
            @test leaf[k] ≈ zg₂[name][k] rtol = T(√eps(T))
        end
    end

    # a seed of the other float type is converted to the parameters', and the gradient keeps their
    # element type
    g_other = back(T === Float32 ? 1.0 : 1.0f0)
    for (name, leaf) in pairs(g_other)
        for k in keys(leaf)
            @test eltype(leaf[k]) === T
            @test leaf[k] ≈ g₁[name][k] rtol = T(√eps(T))
        end
    end
end

# Data of the other float type: the gradient keeps the parameters' element type, as
# `ZygotePullback`'s does, so `train!` can step with it. The two gradients agree to the precision of
# the coarser type.
@testset "data of the other float type gives a gradient of the parameters' type ($T)" for T in REAL_ELTYPES
    S = T === Float32 ? Float64 : Float32
    Random.seed!(26)
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    X = randn(Random.Xoshiro(27), S, 2, 5)
    Y = randn(Random.Xoshiro(28), S, 2, 5)

    value, back = pullback(ps, nn.model, (X, Y))
    g = back(one(value))
    zygote_value, zygote_back = ZygotePullback(HNNLoss(arch))(ps, nn.model, (X, Y))
    zg = zygote_back(one(zygote_value))

    @test g isa NetworkParameters
    for (name, leaf) in pairs(g)
        for k in keys(leaf)
            @test eltype(leaf[k]) === T
            @test eltype(zg[name][k]) === T
            @test leaf[k] ≈ zg[name][k] rtol = √eps(Float32)
        end
    end
end
