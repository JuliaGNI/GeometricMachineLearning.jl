using GeometricMachineLearning
using Test
import Random
Random.seed!(123)

f1(x) = 1cos(x) - 2sin(x) + 1.5cos(2x) - 2.5sin(2x) + 2cos(3x) - 1sin(4x)

nsamples = 1000
xsamples = Float32.(collect(range(0, 5, nsamples)))
ysamples = Float32.(f1.(xsamples))

nwidth = 64
nbatch = 10
nepochs = 2000

model = Chain(Dense(1, nwidth), Dense(nwidth, 1))
nn = NeuralNetwork(model, Float32)
dl = DataLoader(xsamples, ysamples)
o = Optimizer(AdamOptimizer(), nn)
batch = Batch(nbatch, 1, 1)

loss = FeedForwardLoss()

loss_array = o(nn, dl, batch, nepochs, loss)
@test loss_array[end] < 0.9