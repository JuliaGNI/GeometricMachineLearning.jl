using GeometricMachineLearning, LinearAlgebra, ProgressMeter, Plots
import Lux, Zygote, Random, MLDatasets, Flux, Lux.gpu

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]