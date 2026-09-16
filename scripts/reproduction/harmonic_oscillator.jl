using GeometricMachineLearning

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: hamiltonian, default_parameters

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set, which is how the CI
# job runs this script end to end in seconds. See scripts/README.md.
include("../utilities/smoke.jl")

# create the object ensemble_solution
ensemble_problem = hodeensemble(timespan = (0.0, 4.0))
ensemble_solution = exact_solution(ensemble_problem)

include("../utilities/ensemble_plots.jl")

# `GSympNet` trains through the generic `DataLoader` + `Batch` + `Optimizer` pipeline directly,
# as `scripts/sympnets/sympnet_toda_lattice.jl` already does.
dl = DataLoader(ensemble_solution)

arch = GSympNet(dl; n_layers = 4, upscaling_dimension = 10)
nn = NeuralNetwork(arch, Float64)
o = Optimizer(AdamOptimizer(), nn)
batch = Batch(100)
n_epochs = smoke_size(1000, 2)

loss_array = o(nn, dl, batch, n_epochs)

function H(x)
    hamiltonian(
        0.0, x[1:(length(x) ÷ 2)], x[(1 + length(x) ÷ 2):end], default_parameters())
end

plot_result(dl, nn, H; batch_nb_trajectory = 10,
    filename = "GSympNet_4-10_on_Harmonic_Oscillator.png", nb_prediction = 5)

CairoMakie.save("GSympNet_4-10_on_Harmonic_Oscillator_loss.png", plot_loss(loss_array))
