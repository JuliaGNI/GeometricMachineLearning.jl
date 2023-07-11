using GeometricMachineLearning, LinearAlgebra, ProgressMeter, Test
using GeometricMachineLearning: _add
using CUDA

import Lux, Zygote, Random, MLDatasets, Flux, Lux.gpu, KernelAbstractions

backend = CUDA.CUDABackend()

#MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
n_heads = 7
n_layers = 5
patch_number = (image_dim÷patch_length)^2

train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

train_x_reshaped = zeros(Float32, patch_length^2, patch_number, size(train_x, 3))
test_x_reshaped = zeros(Float32, patch_length^2, patch_number, size(test_x, 3))

#preprocessing steps 
for i in 1:size(train_x, 3)
    train_x_reshaped[:, :, i] = split_and_flatten(train_x[:, :, i], patch_length)
end
for i in 1:size(test_x, 3)
    test_x_reshaped[:, :, i] = split_and_flatten(test_x[:, :, i], patch_length)
end

#implement this encoding yourself!
train_y = Flux.onehotbatch(train_y, 0:9) 
test_y = Flux.onehotbatch(test_y, 0:9)

model = MultiHeadAttention(patch_length^2, n_heads)
ps, st = Lux.setup(CUDA.device(), Random.default_rng(), model)

function main(n_data)

    @time output1 = Lux.apply(model, train_x_reshaped[:,:,1:n_data] |> cu, ps, st)[1]

    output2 = KernelAbstractions.zeros(backend, eltype(train_x), patch_length^2, patch_number, n_data)

    @time for i in 1:n_data
        output2[:, :, i] = Matrix(Lux.apply(model, train_x_reshaped[:, :, i] |> cu, ps, st)[1])
    end

    @test isapprox(output1, output2)
end

function main₂(n_data)
    #this first function is for testing purposes only
    #loss(ps) = mapreduce(i -> norm(Lux.apply(model, train_x_reshaped[:,:,1:n_data] |> cu, ps, st)[1][:,:,i]), +, 1:n_data)
    @time output1 = Zygote.gradient(ps -> norm(Lux.apply(model, train_x_reshaped[:,:,1:n_data] |> cu, ps, st)[1]), ps)[1]

    output2 = Zygote.gradient(ps -> norm(Lux.apply(model, train_x_reshaped[:,:,1] |> cu, ps, st)[1]), ps)[1]
    @time for i in 2:n_data
        output2 = _add(output2, Zygote.gradient(ps -> norm(Lux.apply(model, train_x_reshaped[:,:,i] |> cu, ps, st)[1]), ps)[1])
    end

    #=
    for key1 in (Symbol("PQ"), Symbol("PK"), Symbol("PV"))
        for head in 1:n_heads
            key2 = Symbol("head_"*string(head))
            @test isapprox(output1[key1][key2], output2[key1][key2])
        end
    end
    output1, output2
    =#
end


for n_data = 100:100:1000
    print("n_data = ", n_data, "\n")
    main(n_data)
    main₂(n_data)
    print("\n")
end