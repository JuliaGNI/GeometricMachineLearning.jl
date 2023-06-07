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

#this uses a local coordinate representation of the Grassmannian to compute the loss
function grassmann_based_loss(Y::AbstractMatrix, y::AbstractVector)
    norm(reshape(σ.(Y*inv(Y[6:7,1:5])), 10) - y)
end

function grassmann_based_loss(ps::NamedTuple, t)
    grassmann_based_loss(Lux.apply(model, train_x_tuple[t]), train_y[:, t])
end

model = GrassmannLayer(157, 7)
ps, st = Lux.setup(Random.default_rng(), model)

optim = AdamOptimizer()
cache = init_optimizer_cache(optim, model)

function training(n_steps = 100, batch_size=10)
    for step in 1:n_steps
        g = Zygote.gradient(ps -> grassmann_based_loss(ps, batch))
    end
end