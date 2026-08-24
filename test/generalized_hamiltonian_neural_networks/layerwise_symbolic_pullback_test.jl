# The layerwise `SymbolicPullback` on a parameter-dependent generalized HNN.
#
# `SymbolicNeuralNetworks` composes the pullback of a chain layer by layer, putting fresh symbolic
# variables between two layers instead of inlining one layer's expression into the next. Until
# SymbolicNeuralNetworks 0.7 that seam was a plain vector, so a `SymplecticEuler` built with
# `return_parameters = true` -- which threads the parameters of the system on to the next layer, and so
# returns a `Tuple` -- could not be seeded at all (SNN #54). It now says how it meets the seam, which is
# what GML #245 was waiting for.
#
# Two things are worth testing beyond agreement. First, `n_integrators > 1`: the monolithic expression
# grows multiplicatively with it and exceeds 10⁹ terms at two, so nothing built that before. Second, an
# architecture with `parameters`: the monolithic construction traces the chain from a plain vector, so
# the layers default the system parameters away and it does not build at all -- the layerwise
# construction is the only one that is right there, not merely the quicker one.
#
# `Zygote` is the reference throughout. The losses here are not additive over a batch (both normalise
# by `norm(output)` taken over the whole of it), and `SymbolicPullback` sums the per-sample gradients,
# so the comparison is on a single sample -- the same statement SymbolicNeuralNetworks' own suite pins
# down for `FeedForwardLoss`.

using GeometricMachineLearning
using GeometricMachineLearning: ParametricLoss, apply_parametric, _unwrap_gradient
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: composes_layerwise, symbolic_steps, layerwise_gradient_function,
                              loss_seed, checked_layer_seed, layer_seed
using AbstractNeuralNetworks: FeedForwardLoss, params, layers
using NeuralNetworkParameters: flatten
using LinearAlgebra: norm
using Random: seed!
using Test
import Zygote

seed!(1234)

# The gradients nest -- a `SymplecticEuler` holds the parameters of a whole energy sub-network -- so
# they are compared through their flat form rather than key by key.
flat(gradient) = flatten(_unwrap_gradient(gradient))[1]
maximum_difference(a, b) = maximum(abs, flat(a) - flat(b))
construction(pullback) = typeof(pullback.fun.gradient_function).name.name

input, output = rand(4, 1), rand(4, 1)
system_parameters = (m = 1.0, ω = π / 2)

@testset "the chain decomposes and every layer can now be seeded: $n integrator(s)" for n in (1, 2)
    arch = GeneralizedHamiltonianArchitecture(4; width = 4, nhidden = 1, n_integrators = n)
    snn = SymbolicNeuralNetwork(arch)
    sparams = params(snn)

    # two steps per integrator: one `SymplecticEulerA`, one `SymplecticEulerB`
    @test length(symbolic_steps(snn)) == 2n
    @test composes_layerwise(snn)
    # all but the last thread the system parameters on, and used to be the ones that could not be seeded
    for (layer, key) in symbolic_steps(snn)
        @test !isnothing(checked_layer_seed(layer, key, sparams[key]))
    end
    @test !isnothing(layerwise_gradient_function(snn, FeedForwardLoss()))
end

@testset "a parameter-free GHNN agrees with the monolithic path and with Zygote" begin
    arch = GeneralizedHamiltonianArchitecture(4; width = 4, nhidden = 1, n_integrators = 1)
    snn, nn = SymbolicNeuralNetwork(arch), NeuralNetwork(arch, Float64)
    loss = FeedForwardLoss()

    layerwise = SymbolicNeuralNetworks.SymbolicPullback(snn, loss; layerwise = :auto)
    monolithic = SymbolicNeuralNetworks.SymbolicPullback(snn, loss; layerwise = false)
    # `:auto` used to throw here, on a network the monolithic path builds without trouble
    @test construction(layerwise) == :LayerwiseGradientFunction
    @test construction(monolithic) != :LayerwiseGradientFunction

    gradient = layerwise.fun(input, output, params(nn))(1)
    @test maximum_difference(gradient, monolithic.fun(input, output, params(nn))(1)) < 1e-14
    reference = Zygote.gradient(p -> loss(nn.model, p, input, output), params(nn))[1]
    @test maximum_difference(gradient, reference) < 1e-14
end

# The expression of two integrators has about 10⁹ terms, so this is the first construction that reaches
# it at all.
@testset "two integrators build, which no monolithic expression does" begin
    arch = GeneralizedHamiltonianArchitecture(4; width = 4, nhidden = 1, n_integrators = 2)
    snn, nn = SymbolicNeuralNetwork(arch), NeuralNetwork(arch, Float64)
    loss = FeedForwardLoss()

    pullback = SymbolicNeuralNetworks.SymbolicPullback(snn, loss; layerwise = true)
    @test length(pullback.fun.gradient_function.steps) == 4
    # the sweep never asks the first layer for the sensitivity to its input
    @test isnothing(first(pullback.fun.gradient_function.steps).dλ)

    reference = Zygote.gradient(p -> loss(nn.model, p, input, output), params(nn))[1]
    @test maximum_difference(pullback.fun(input, output, params(nn))(1), reference) < 1e-14

    # GML's own two constructors still build one expression for the whole network, so their limit
    # stands and now says where to go instead. `SymbolicPullback(arch)` uses `HNNLoss`, which
    # evaluates the pre-built vector field rather than the chain, so there is no prediction for a
    # layerwise seed to start from at all.
    raised = try
        SymbolicPullback(arch)
    catch error
        error
    end
    @test raised isa ArgumentError
    @test occursin("composes the pullback layer by layer", raised.msg)
end

@testset "a parametrized GHNN: the carried system parameters reach the gradient" begin
    arch = GeneralizedHamiltonianArchitecture(4; width = 4, nhidden = 1, n_integrators = 1,
                                              parameters = system_parameters)
    snn, nn = SymbolicNeuralNetwork(arch), NeuralNetwork(arch, Float64)
    loss = FeedForwardLoss()

    pullback = SymbolicNeuralNetworks.SymbolicPullback(snn, loss; layerwise = true)
    @test construction(pullback) == :LayerwiseGradientFunction

    gradient = pullback.fun((input, system_parameters), output, params(nn))(1)
    reference = Zygote.gradient(params(nn)) do p
        norm(apply_parametric(nn.model, input, system_parameters, p) - output) / norm(output)
    end[1]
    @test maximum_difference(gradient, reference) < 1e-14

    # the monolithic construction cannot be built for this architecture at all: it traces the chain
    # from a plain vector, and the layers then default the system parameters away, leaving the energy
    # network's first `Dense` short of the components it reads
    @test_throws Exception SymbolicNeuralNetworks.SymbolicPullback(snn, loss; layerwise = false)
end

# `ParametricLoss` is the loss the parameter-dependent architectures train with, and it takes the
# system parameters as a fifth argument -- so the layerwise construction cannot guess its expression
# and it declares one instead.
@testset "ParametricLoss declares its expression, so the sweep can seed it" begin
    arch = GeneralizedHamiltonianArchitecture(4; width = 4, nhidden = 1, n_integrators = 2,
                                              parameters = system_parameters)
    snn, nn = SymbolicNeuralNetwork(arch), NeuralNetwork(arch, Float64)
    loss = ParametricLoss()

    @test !isnothing(SymbolicNeuralNetworks.loss_expression(loss, [1.0], [1.0]))
    @test !isnothing(loss_seed(loss, snn))

    pullback = SymbolicNeuralNetworks.SymbolicPullback(snn, loss; layerwise = :auto)
    @test construction(pullback) == :LayerwiseGradientFunction

    gradient = pullback.fun((input, system_parameters), output, params(nn))(1)
    reference = Zygote.gradient(p -> loss(nn.model, p, input, output, system_parameters),
                                params(nn))[1]
    @test maximum_difference(gradient, reference) < 1e-14
end
