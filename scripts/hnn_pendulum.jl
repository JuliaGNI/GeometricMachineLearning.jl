# using Profile
#
# NOTE: this script does not run to completion, and what stops it is in the library rather than
# here. Every training method for a Hamiltonian neural network — `ExactHnn`, which `default_method`
# picks for these data, and `SEulerA`/`SEulerB` — calls `vectorfield(nn, x, params)` at
# `src/training_method/hnn_exact_method.jl:6` and `src/training_method/symplectic_euler.jl:13,18`,
# and no such method exists: `vectorfield` resolves only to GeometricBase's methods on
# `AbstractStateVariable` and `State`. HNN training through `train!` is therefore unavailable for
# every method, independently of this script. See *Open Issues → B6* in `CHANGELOG.md`.
using GeometricMachineLearning

# this contains the functions for generating the training data
include("pendulum.jl")

# this contains the functions for generating the plots
# include("plots.jl")

# layer dimension/width
const ld = 5

# hidden layers
const ln = 3

# number of inputs/dimension of system
const ninput = 2

# number of training runs
const nruns = 1000

# Optimiser. The argument is the momentum coefficient; the step size belongs to the `Optimizer`
# rather than to the method, so it is passed to `train!` below.
#opt = GradientOptimizer()
opt = MomentumOptimizer(0.5)

# create HNN
# The constructor takes `(dim, width, nhidden, activation)` positionally.
# `StandardHamiltonianArchitecture` is named explicitly, because calling the abstract
# `HamiltonianArchitecture` warns and defaults to it.
hnn = StandardHamiltonianArchitecture(ninput, ld, ln)

# create Lux network
nn = NeuralNetwork(hnn, CPU(), Float64)

# get data set
training_data = get_data_set()

# perform training (returns array that contains the total loss for each training step)
total_loss = train!(nn, training_data, opt; ntraining = nruns, step_size = 1e-2)

#time training (after warmup)
# total_loss = train!(hnn, data, target; ntraining = nruns, learning_rate = η)
# @time total_loss = train!(hnn, data, target; ntraining = nruns, learning_rate = η)

#profile training
#run with julia --track-allocation=user hnn.jl
# Profile.clear()
# Profile.clear_malloc_data()
# @profile total_loss = train!(hnn, data, target; ntraining = nruns, learning_rate = η)

# plot results
include("plots.jl")
plot_hnn(H, nn, total_loss; filename = "hnn_pendulum.png")
