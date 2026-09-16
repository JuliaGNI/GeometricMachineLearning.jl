# Train a Hamiltonian neural network on the pendulum and plot what it learned.
using GeometricMachineLearning

# this contains the Hamiltonian and the functions for generating the training data
include("../utilities/pendulum.jl")

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set. See scripts/README.md.
include("../utilities/smoke.jl")

# layer dimension/width
const ld = 5

# hidden layers
const ln = 3

# number of inputs/dimension of system
const ninput = 2

# number of epochs
const nepochs = smoke_size(2000, 2)

# `StandardHamiltonianArchitecture` is named explicitly, because calling the abstract
# `HamiltonianArchitecture` warns and defaults to it. The constructor takes
# `(dim, width, nhidden, activation)` positionally.
const hnn = StandardHamiltonianArchitecture(ninput, ld, ln)

nn = NeuralNetwork(hnn, CPU(), Float64)

# A grid of phase-space points and the symplectic gradient at each.
input, output = get_data_set()
dl = DataLoader(input, output)

# `HNNLoss` asks that the Hamiltonian vector field of the learned Hamiltonian match the vector field
# in the data. It is the loss `NetworkLoss(hnn)` returns, so naming it is for the reader.
loss = HNNLoss(hnn)

# The step size belongs to the `Optimizer`, not to the method: the same `Adam()` trains at any
# learning rate. `1e-3` is already Adam's default here and is named for the reader.
optimizer = Optimizer(Adam(), nn; step_size = 1e-3)
total_loss = optimizer(nn, dl, Batch(10), nepochs, loss; show_progress = true)

# plot results
include("../utilities/plots.jl")
plot_hnn(H, x -> only(nn(x)), total_loss; filename = "hnn_pendulum.png")
