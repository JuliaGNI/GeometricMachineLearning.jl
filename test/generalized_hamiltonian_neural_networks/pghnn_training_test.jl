# One epoch of training a `GeneralizedHamiltonianArchitecture` on a `ParametricDataLoader`, which is
# the path that ties the pieces together: the batch splitter, the parametric loss, the `Zygote`
# pullback through the symbolic gradient of the energies, and the optimizer step over a *nested*
# parameter set.

using GeometricMachineLearning
using AbstractNeuralNetworks: params
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate
using Random: seed!
using Test

seed!(1234)

function shift_parameters(params::NamedTuple, a::Number)
    NamedTuple{keys(params)}(Tuple(value .+ a for value in values(params)))
end

all_parameters = [default_parameters(), shift_parameters(default_parameters(), 0.5)]

sol = integrate(hodeensemble(; parameters = all_parameters), ImplicitMidpoint())
dl = ParametricDataLoader(sol)

arch = GeneralizedHamiltonianArchitecture(dl.input_dim; parameters = default_parameters())
nn = NeuralNetwork(arch)
parameters_before = deepcopy(params(nn))

n_epochs = 2
loss_array = Optimizer(AdamOptimizer(), nn)(nn, dl, Batch(200), n_epochs; show_progress = false)

@test length(loss_array) == n_epochs
@test all(isfinite, loss_array)
@test all(>(0), loss_array)

# The optimizer has to reach every block of the *nested* parameter set: the architecture is a chain
# of `SymplecticEuler` layers, each of which holds the parameters of a whole sub-network.
function every_block_moved(before, after)
    all(keys(before)) do layer
        all(keys(before[layer])) do sublayer
            all(keys(before[layer][sublayer])) do parameter
                before[layer][sublayer][parameter] != after[layer][sublayer][parameter]
            end
        end
    end
end

@test every_block_moved(parameters_before, params(nn))
