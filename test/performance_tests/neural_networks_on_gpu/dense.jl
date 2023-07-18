using Lux 
using CUDA

import Random

model_size = 100
n_data = 1000

model = Dense(model_size, 1, tanh)

data = Tuple([rand(model_size) for i in 1:n_data])

function one_layer_neural_network(x, a, b)
    a'*x + b
end 

function cpu(n_steps, data)
    ps, st = Lux.setup(Random.default_rng(), model)
    loss(x, ps, st) = sum(Lux.apply(model, x, ps, st)[1])
    loss_val = 0 
    for i in 1:n_data 
        x_val = data[i]
        loss_val += loss(x_val, ps, st)
    end
    loss_val
end 

function gpu(n_steps, data)
    data = Tuple([data[i] |> cu for i in 1:n_data])

    ps, st = Lux.setup(Random.default_rng(), model) |> cu
    store_loss = CUDA.zeros(n_data)

    function kernel(loss_array, x)
        loss_array[threadIdx().x] = sum(Lux.apply(model, x, ps, st)[1])
        return 
    end

    dat_temp = data[1]
    @cuda threads=n_data kernel(store_loss, dat_temp)
    sum(store_loss)
end

function gpu₂(n_steps, data)
    data = Tuple([data[i] |> cu for i in 1:n_data])

    a = CUDA.rand(model_size)
    b = rand(Float32)
    store_loss = CUDA.zeros(n_data)

    function kernel!(loss_array, x, a, b)
        loss_array[threadIdx().x] = one_layer_neural_network(x, a, b)
        return nothing
    end

    dat_temp = data[1]
    @cuda threads=n_data kernel!(store_loss, dat_temp, a, b)
    sum(store_loss)
end