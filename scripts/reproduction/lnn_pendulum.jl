# Train a Lagrangian neural network on the pendulum.
#
# The data are `(q, q̇)` on a grid and the acceleration the *exact* Lagrangian implies at each point.
# `LNNLoss` solves the Euler-Lagrange equations of the *learned* Lagrangian for its own acceleration
# and compares the two, so a decreasing loss means the network is recovering a Lagrangian whose
# dynamics match the pendulum's.
using GeometricMachineLearning

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set, which is how the CI
# job runs this script end to end in seconds. See scripts/README.md.
include("../utilities/smoke.jl")

# this contains the Lagrangians and the functions for generating the training data
include("../utilities/data_problem.jl")

const problem = :pendulum

# Layer dimension/width, and the number of hidden layers. These are the sizes that cost time here,
# and `nepochs` is not: `LNNLoss` reaches the Euler-Lagrange acceleration through a *symbolic*
# gradient and Hessian of the chain, and the run's wall clock is the one-time compilation of their
# pullback, which grows with the chain and falls in the first epoch -- in a smoke run 106s at
# `(5, 3)`, 49s at `(5, 1)` and 21s at `(2, 1)`, against 1.8s for a second epoch. The smoke sizes
# below are `LagrangianNeuralNetwork`'s own defaults, `width = dimin` and `nhidden = 1` -- the
# smallest chain the constructor offers, through the same symbolic path as the full size.
const ld = smoke_size(5, 2)
const ln = smoke_size(3, 1)

const _, n_dim = dict_problem_L[problem]

# number of inputs/dimension of system
const ninput = 2 * n_dim

# number of epochs
const nepochs = smoke_size(200, 2)

const arch = LagrangianNeuralNetwork(ninput; nhidden = ln, width = ld)

nn = NeuralNetwork(arch, CPU(), Float64)

input, output = get_LNN_data(problem)
dl = DataLoader(input, output)

# `LNNLoss(arch)` is what `NetworkLoss(arch)` returns; naming it is for the reader.
loss = LNNLoss(arch)

# The step size belongs to the `Optimizer`, not to the method. `LNNLoss` needs a larger one than
# `HNNLoss` does: its gradient reaches the parameters through a solve against the velocity Hessian,
# where `HNNLoss` is linear in the gradient of the learned Hamiltonian.
optimizer = Optimizer(Adam(), nn; step_size = 1e-2)
total_loss = optimizer(nn, dl, Batch(10), nepochs, loss; show_progress = true)

# plot results
include("../utilities/plots.jl")
L, _ = dict_problem_L[problem]
plot_hnn(L, x -> only(nn(x)), total_loss;
    filename = "lnn_pendulum.png", xmin = -1.2, xmax = +1.2, ymin = -1.2, ymax = +1.2)
