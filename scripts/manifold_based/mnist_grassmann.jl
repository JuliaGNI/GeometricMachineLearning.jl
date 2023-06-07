using GeometricMachineLearning, LinearAlgebra, ProgressMeter, Plots
import Lux, Zygote, Random, MLDatasets, Flux, Lux.gpu

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

function reshape_x(x::AbstractMatrix{T}) where {T}
    x_new = reshape(vcat(reshape(x, 28*28), T(0.)), 157, 5)
end

train_x_tuple = map(_ -> zeros(Float32, 157, 5), 1:size(train_x, 3)) |> Tuple
test_x_tuple = map(_ -> zeros(Float32, 157, 5), 1:size(test_x, 3)) |> Tuple

for t in 1:size(train_x, 3)
    train_x_tuple[t] .= reshape_x(train_x[:, :, t])
end

for t in 1:size(test_x, 3)
    test_x_tuple[t] .= reshape_x(test_x[:, :, t])
end

#implement this encoding yourself!
train_y = Flux.onehotbatch(train_y, 0:9) #|> gpu
test_y = Flux.onehotbatch(test_y, 0:9) #|> gpu

#model = GrassmannLayer()