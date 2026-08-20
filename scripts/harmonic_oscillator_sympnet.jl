using GeometricMachineLearning
using GeometricIntegrators: ImplicitMidpoint, integrate
import GeometricProblems.HarmonicOscillator as ho

# the problem is the ODE of the harmonic oscillator
ho_problem = ho.hodeproblem(; tspan = 500)

# integrate the system
solution = integrate(ho_problem, ImplicitMidpoint())

dl_raw = DataLoader(solution; suppress_info = true)

# specify the data type and the backend
type = Float64
backend = CPU()

# we can then make a new instance of `DataLoader` with this backend and type.
dl = DataLoader(dl_raw, backend, type)


const upscaling_dimension = 2
const nhidden = 1
const activation = tanh
const n_layers = 4 # number of layers for the G-SympNet
const depth = 4 # number of layers in each linear block in the LA-SympNet

# calling G-SympNet architecture
gsympnet = GSympNet(dl; upscaling_dimension = upscaling_dimension,
                        n_layers = n_layers,
                        activation = activation)

# initialize the networks
g_nn = NeuralNetwork(gsympnet, backend, type)

# set up optimizer; for this we first need to specify the optimization method
opt_method = AdamOptimizer(type)

# we then call the optimizer struct which allocates the cache
g_opt = Optimizer(opt_method, g_nn)

# determine the batch size (the number of samples in one batch)
const batch_size = 16

batch = Batch(batch_size)

# number of training epochs
const nepochs = 100

# perform training (returns array that contains the total loss for each training step)
g_loss_array = g_opt(g_nn, dl, batch, nepochs; show_progress = false)

ics = (q=dl.input.q[:, 1, 1], p=dl.input.p[:, 1, 1])

steps_to_plot = 1000

#predictions
g_trajectory =  iterate(g_nn, ics; n_points = steps_to_plot)

