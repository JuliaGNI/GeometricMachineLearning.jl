#import module
using GeometricMachineLearning

# Import data
include("data_problem.jl")

nameproblem = :pendulum
L,n_dim = dict_problem_L[nameproblem]

println("Begin generating data")

Data, Target = get_LNN_data(nameproblem)

Get_Data = Dict(
    :nb_points => Data -> length(Data),
    :q => (Data,n) -> Data[n][1][1],
    :q̇ => (Data,n) -> Data[n][2][1]
)
pdata = data_sampled(Data, Get_Data)

Get_Target = Dict(
    :q̈ => (Target,n) -> Target[n][1],
)

data = dataTarget(pdata, Target, Get_Target)

println("End generating data")

# layer dimension/width
const ld = 5

# hidden layers
const ln = 1

# number of inputs/dimension of system
const ninput = n_dim*2

# number of training runs
const nruns = 1000

# create HNN
lnn = LagrangianNeuralNetwork(ninput; nhidden = ln, width = ld)

# create Lux network
nn = NeuralNetwork(lnn, LuxBackend())

#training method
hti = ExactIntegratorLNN()

# Optimiser
opt = MomentumOptimizer(1e-3,0.5)


# perform training (returns array that contains the total loss for each training step)
total_loss = train!(nn, opt, data; ntraining = nruns, lti = hti)

# plot results
include("plots.jl")
plot_hnn(L, nn, total_loss; filename="lnn_pendulum.png", xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2)


