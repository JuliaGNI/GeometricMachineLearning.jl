# Train a Lagrangian neural network on the pendulum.
#
# The data are `(q, q̇)` on a grid and the acceleration the *exact* Lagrangian implies at each point.
# `LNNLoss` solves the Euler-Lagrange equations of the *learned* Lagrangian for its own acceleration
# and compares the two, so a decreasing loss means the network is recovering a Lagrangian whose
# dynamics match the pendulum's.
using GeometricMachineLearning

# this contains the Lagrangians and the functions for generating the training data
include("data_problem.jl")

const problem = :pendulum

# layer dimension/width
const ld = 5

# hidden layers
const ln = 3

const _, n_dim = dict_problem_L[problem]

# number of inputs/dimension of system
const ninput = 2 * n_dim

# number of epochs
const nepochs = 200

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
include("plots.jl")
L, _ = dict_problem_L[problem]
plot_hnn(L, x -> only(nn(x)), total_loss;
    filename = "lnn_pendulum.png", xmin = -1.2, xmax = +1.2, ymin = -1.2, ymax = +1.2)
