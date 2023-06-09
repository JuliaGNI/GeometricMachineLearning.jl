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

#Zygote has trouble differentiating indexing opeartions
projection₁ = hcat(I(5), zeros(Float32, 5, 2))
projection₂ = hcat(zeros(Float32, 2, 5), I(2))

function grassmann_based_loss(Y::AbstractMatrix, y::AbstractVector)
    norm(projection₂*Y - reshape(y, 2, 5)*projection₁*Y)
end

function grassmann_based_loss(ps::NamedTuple, x::AbstractMatrix, y::AbstractVector)
    grassmann_based_loss(Lux.apply(model, x, ps, st)[1], y)
end

model = Lux.Chain(GrassmannLayer(157, 7, Transpose=true), Lux.Dense(7, 7, use_bias=false))
ps, st = Lux.setup(Random.default_rng(), model)

optim = AdamOptimizer()
cache = init_optimizer_cache(model, optim)

#optim.t = 1

function training(n_steps = 500, batch_size=10)
    loss_array = zeros(n_steps)
    @showprogress for step in 1:n_steps
        batch₁ = Int(ceil(length(train_x_tuple)*rand()))
        x₁ = train_x_tuple[batch₁]
        y₁ = train_y[:, batch₁]
        loss_and_gradient = Zygote.withgradient(ps -> grassmann_based_loss(ps, x₁, y₁), ps)
        loss = loss_and_gradient[1]
        grad = loss_and_gradient[2][1]
        for _ in 2:batch_size
            batch = Int(ceil(size(train_x, 3)*rand()))
            x₁ = train_x_tuple[batch]
            y₁ = train_y[:, batch]
            loss_and_gradient = Zygote.withgradient(ps -> grassmann_based_loss(ps, x₁, y₁), ps)
            loss += loss_and_gradient[1]
            add!(grad, grad, loss_and_gradient[2][1])
        end
        loss_array[step] = loss
        optimization_step!(optim, model, ps, cache, grad)
    end
    loss_array
end

loss_array = training()