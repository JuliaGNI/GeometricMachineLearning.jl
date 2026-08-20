# The symbolic pullback of a parameter-dependent network, on the smallest such network there is: a
# single `SymplecticEulerB` layer built from a `SymbolicPotentialEnergy`.
#
# `SymbolicPullback(nn, ::ParametricLoss, system_params)` builds its gradient with `reduce = +`, so
# what it returns is the *sum* of the per-sample gradients -- that is the convention
# `SymbolicNeuralNetworks.SymbolicPullback` uses too. The test compares against exactly that,
# computed with `Zygote` one sample at a time.

using GeometricMachineLearning
using GeometricMachineLearning: ParametricLoss, SymbolicPotentialEnergy, SymplecticEulerB
using AbstractNeuralNetworks: params
using Random: seed!
using Test
import Zygote

seed!(1234)

system_parameters = (m = 1.0, ω = π / 2)
dim, width, nhidden, activation = 2, 2, 1, tanh
n_samples = 10

se = SymbolicPotentialEnergy(dim, width, nhidden, activation; parameters = system_parameters)
nn = NeuralNetwork(Chain(SymplecticEulerB(se; return_parameters = false)))

loss = ParametricLoss()
pullback = SymbolicPullback(nn, loss, system_parameters)

input = rand(dim, n_samples)
output = rand(dim, n_samples)
# one parameter set per sample, which is the shape `ParametricDataLoader` hands to the optimizer
batch_parameters = fill(system_parameters, n_samples)

loss_value, gradient = pullback(params(nn), nn.model, (input, output, batch_parameters))

@test loss_value ≈ loss(nn.model, params(nn), input, output, batch_parameters)

symbolic_gradient = gradient(1.0)

function summed_per_sample_gradient()
    total = nothing
    for i in 1:n_samples
        single = Zygote.gradient(
            ps -> loss(nn.model, ps, input[:, i:i], output[:, i:i], [system_parameters]),
            params(nn))[1]
        block = single.L1.params
        total = isnothing(total) ? block : map((a, b) -> map(+, a, b), total, block)
    end
    total
end

reference_gradient = summed_per_sample_gradient()

@test keys(symbolic_gradient) == (:L1,)
for layer in keys(reference_gradient)
    for parameter in keys(reference_gradient[layer])
        @test symbolic_gradient.L1[layer][parameter] ≈ reference_gradient[layer][parameter]
    end
end

# A gradient of all zeros would pass the loop above if the reference were zero too, so check that
# the network actually depends on its parameters here.
@test any(any(abs.(block) .> 1e-8) for layer in keys(reference_gradient)
          for block in values(reference_gradient[layer]))

# Building the pullback for a stacked architecture is refused rather than left to hang: the symbolic
# expression grows multiplicatively with `n_integrators` and already exceeds 10⁹ terms at two.
stacked = NeuralNetwork(GeneralizedHamiltonianArchitecture(dim; n_integrators = 2,
                                                           parameters = system_parameters))
@test_throws ArgumentError SymbolicPullback(stacked, loss, system_parameters)

# one integrator is the supported case, and it builds
single = NeuralNetwork(GeneralizedHamiltonianArchitecture(dim; n_integrators = 1,
                                                          parameters = system_parameters))
@test SymbolicPullback(single, loss, system_parameters) isa SymbolicPullback

