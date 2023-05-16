# using Profile
using GeometricMachineLearning

# this contains the functions for generating the training data
include("pendulum.jl")

# this contains the functions for generating the plots
# include("plots.jl")

# layer dimension/width
const ld = 2

# hidden layers
const ln = 1

# number of inputs/dimension of system
const ninput = 2

# learning rate
const η = .001

# number of training runs
const nruns = 1000

# Optimiser
opt = GradientOptimizer(1e-2)


# create HNN
hnn = HamiltonianNeuralNetwork(ninput; nhidden = ln, width = ld)

# create Lux network
nn = NeuralNetwork(hnn, LuxBackend())

# get data set
data, target = get_data_set()

# perform training (returns array that contains the total loss for each training step)
total_loss = train!(nn, opt, data, target; ntraining = nruns)

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
plot_hnn(H, nn, total_loss; filename="hnn_pendulum.png")



#=test

pp = nn.params

p = Tuple([Tuple(x) for x in pp])

keys_1 = keys(pp)
keys_2 = [keys(x) for x in values(pp)]

ppr = NamedTuple(zip(keys_1,[NamedTuple(zip(k,x)) for (k,x) in zip(keys_2,p)]))



f(x) = nn([1,2],x)
df(x) = Zygote.gradient(f,x)[1]

=#