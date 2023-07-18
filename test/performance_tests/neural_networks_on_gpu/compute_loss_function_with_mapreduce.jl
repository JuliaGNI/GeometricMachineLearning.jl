using Lux 
using CUDA
using LinearAlgebra 
import Random

function main(N, n_data)

    data = Tuple(rand(Float32, N) for i in 1:n_data)

    model = Chain(Dense(N, N, tanh), Dense(N, 1, Lux.σ))

    loss(x, ps, st) = norm(Lux.apply(model, x, ps, st)[1])

    ps, st = Lux.setup(Random.default_rng(), model)

    function cpu_performance(N, n_data)
        @time "cpu:" loss_total = mapreduce(i -> loss(data[i], ps, st), +, 1:n_data)

        #print("total loss = ",loss_total, "\n")
    end

    function gpu_performance(N, n_data)
        ps = ps |> cu
        data_gpu = Tuple(data[i] |> cu for i in 1:n_data)

        @time "gpu:" loss_total = mapreduce(i -> loss(data_gpu[i], ps, st), +, 1:n_data)

        #print("total loss = ",loss_total, "\n")
    end

    cpu_performance(N, n_data)
    gpu_performance(N, n_data)
end 

for N in 2 .^(8:10)
    for n_data in 2 .^(10:17)
        print("N = ", N,  " and data size is: ", n_data, "\n")
        main(N, n_data)
    end
    print("\n")
end