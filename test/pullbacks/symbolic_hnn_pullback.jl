using GeometricMachineLearning
using GeometricMachineLearning: HNNLoss, ZygotePullback, SigmoidActivation
using AbstractNeuralNetworks: NeuralNetwork, params, model
using LinearAlgebra: norm
using NeuralNetworkParameters: NetworkParameters
using Test
import Random

include("../helpers/eltypes.jl")

# About 1.8e19 in `Float32` and 1.8e154 in `Float64`: the scale at which the product of the two
# norms overflows and each norm does not.
extreme_scale(::Type{Float32}) = Float32(1e20)
extreme_scale(::Type{Float64}) = Float64(1e155)

# The symbolic pullback is the gradient of `HNNLoss` on the batch: the generated per-sample
# gradients of the surrogate `½‖rᵢ‖²`, scaled by the one scalar the batch loss divides by. Two batch
# sizes, because the scale is one only at N = 1 — a batch of one sample is the per-sample sum, and
# the defect the entry recorded was invisible there.
@testset "SymbolicPullback of a Hamiltonian network is HNNLoss's gradient ($T)" for T in REAL_ELTYPES
    for N in (1, 5)
        Random.seed!(11)
        arch = StandardHamiltonianArchitecture(2, 4, 1, SigmoidActivation())
        nn = NeuralNetwork(arch, CPU(), T)
        ps = params(nn)
        X = rand(T, 2, N)
        Y = rand(T, 2, N)

        value, back = SymbolicPullback(arch)(ps, nn.model, (X, Y))
        g = back(one(value))
        zygote_value, zygote_back = ZygotePullback(HNNLoss(arch))(
            ps, nn.model, (X, Y))
        zg = zygote_back(one(zygote_value))

        @test value isa T
        @test value ≈ zygote_value rtol = T(√eps(T))
        @test g isa NetworkParameters
        @test keys(g) == keys(ps)
        for (name, leaf) in pairs(g)
            @test all(k -> eltype(leaf[k]) === T, keys(leaf))
            for k in keys(leaf)
                @test leaf[k] ≈ zg[name][k] rtol = T(√eps(T))
            end
        end

        # the value is `HNNLoss` itself, so a wrong scale cannot hide behind a different value
        hvf = GeometricMachineLearning.hamiltonian_vector_field(arch)
        @test value ≈ norm(hvf(X, ps) - Y) / norm(Y) rtol = T(√eps(T))
    end
end

# The two norms are divided by separately: their product overflows where neither does, and `inv` of
# the product is then a zero gradient. The scale is not a power of two, so round-off shows.
@testset "the scale divides by each norm ($T)" for T in REAL_ELTYPES
    Random.seed!(13)
    arch = StandardHamiltonianArchitecture(2, 4, 1, SigmoidActivation())
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    X = rand(T, 2, 5)
    Y = rand(T, 2, 5) .* extreme_scale(T)

    value, back = SymbolicPullback(arch)(ps, nn.model, (X, Y))
    g = back(one(value))
    zygote_value, zygote_back = ZygotePullback(HNNLoss(arch))(ps, nn.model, (X, Y))
    zg = zygote_back(one(zygote_value))

    @test isfinite(value)
    @test value ≈ zygote_value rtol = T(√eps(T))
    for (name, leaf) in pairs(g)
        for k in keys(leaf)
            @test leaf[k] ≈ zg[name][k] rtol = T(√eps(T))
        end
    end
end

# Data the network's own vector field generates: the residual is zero, the value is zero, and the
# gradient is the zero set rather than `NaN` from an unguarded `inv`.
@testset "a zero residual gives a zero gradient ($T)" for T in REAL_ELTYPES
    Random.seed!(17)
    arch = StandardHamiltonianArchitecture(2, 4, 1, SigmoidActivation())
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    X = rand(T, 2, 5)
    Y = GeometricMachineLearning.hamiltonian_vector_field(arch)(X, ps)

    value, back = SymbolicPullback(arch)(ps, nn.model, (X, Y))
    g = back(one(value))

    @test value == zero(T)
    for (name, leaf) in pairs(g)
        for k in keys(leaf)
            @test all(iszero, leaf[k])
        end
    end
end

# The gradient is linear in the seed, as `ZygotePullback`'s is.
@testset "the seed scales the gradient ($T)" for T in REAL_ELTYPES
    Random.seed!(19)
    arch = StandardHamiltonianArchitecture(2, 4, 1, SigmoidActivation())
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    X = rand(T, 2, 5)
    Y = rand(T, 2, 5)

    value, back = SymbolicPullback(arch)(ps, nn.model, (X, Y))
    _, zygote_back = ZygotePullback(HNNLoss(arch))(ps, nn.model, (X, Y))

    g₁ = back(one(value))
    g₂ = back(T(2))
    zg₂ = zygote_back(T(2))

    for (name, leaf) in pairs(g₂)
        for k in keys(leaf)
            @test leaf[k] ≈ 2 .* g₁[name][k] rtol = T(√eps(T))
            @test leaf[k] ≈ zg₂[name][k] rtol = T(√eps(T))
        end
    end
end
