# Train a Hamiltonian neural network on the pendulum, then *integrate* the vector field it learned
# and compare the trajectory and the energy drift against the true Hamiltonian.
#
# This is the half of the HNN story that `hnn_pendulum.jl` does not tell. That script plots what the
# learned Hamiltonian looks like on a grid; this one asks what happens when you step with it. The
# two questions come apart: a network can match `H` closely on the training grid and still drift in
# energy over a long integration, and no test in the suite checks the second.
using GeometricMachineLearning
using GeometricIntegrators: ODEProblem, ImplicitMidpoint, integrate

# the Hamiltonian, its symplectic gradient, and the training data
include("../utilities/pendulum.jl")

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set. See scripts/README.md.
include("../utilities/smoke.jl")

const ld = 5
const ln = 3
const ninput = 2
const nepochs = smoke_size(2000, 2)

# The integration the comparison is about. `n_steps` stays the same in a smoke run -- stepping is
# cheap next to training, and a handful of steps would show no drift to compare.
const timestep = 0.1
const n_steps = 100
const x₀ = [0.0, 1.0]

const hnn = StandardHamiltonianArchitecture(ninput, ld, ln)
nn = NeuralNetwork(hnn, CPU(), Float64)

input, output = get_data_set()
dl = DataLoader(input, output)

# `HNNLoss` holds the Hamiltonian vector field it compares against the data. Naming the loss keeps
# that field for the integration below, where constructing a second `HNNLoss` would build the
# symbolic derivative again.
const loss = HNNLoss(hnn)

optimizer = Optimizer(Adam(), nn; step_size = 1e-3)
total_loss = optimizer(nn, dl, Batch(10), nepochs, loss; show_progress = true)

# The learned Hamiltonian and the vector field built from it. `loss.hvf` is what `HNNLoss` compares
# against the data, so this integrates exactly the quantity that was trained -- the same object,
# not a second derivative taken again here.
H̃(x) = only(nn(x))
dH̃(x) = loss.hvf(x, nn.params)

# `ODEProblem` calls its vector field as `v(v, t, q, params)`.
reference_field(v, t, q, params) = (v .= dH(q))
learned_field(v, t, q, params) = (v .= dH̃(q))

const timespan = (0.0, n_steps * timestep)
sol_ref = integrate(ODEProblem(reference_field, timespan, timestep, x₀), ImplicitMidpoint())
sol_hnn = integrate(ODEProblem(learned_field, timespan, timestep, x₀), ImplicitMidpoint())

include("../utilities/plots.jl")
plot_network_sim(
    H, H̃, sol_ref, sol_hnn, total_loss; filename = "hnn_pendulum_simulation.png")
