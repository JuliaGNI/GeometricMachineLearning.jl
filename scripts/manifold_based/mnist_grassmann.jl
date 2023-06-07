using GeometricMachineLearning, LinearAlgebra, ProgressMeter, Plots
import Lux, Zygote, Random, MLDatasets, Flux, Lux.gpu

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

#implement this encoding yourself!
train_y = Flux.onehotbatch(train_y, 0:9) #|> gpu
test_y = Flux.onehotbatch(test_y, 0:9) #|> gpu

#model = GrassmannLayer()