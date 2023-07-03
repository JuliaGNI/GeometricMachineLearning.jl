using Lux, Plots, LinearAlgebra, ProgressMeter, CUDA
using GeometricMachineLearning, Random, Zygote

number_data_points = 10000
data_input = 2*π*(1:number_data_points)/number_data_points
data_output = sin.(data_input)

data_input = Tuple([data_input[i]] for i in 1:number_data_points)
data_output = Tuple([data_output[i]] for i in 1:number_data_points)

layer_width = 100
model = Chain(Dense(1, layer_width, tanh), Dense(layer_width, layer_width, tanh), Dense(layer_width, 1))

function train(dev::Device, n_steps, batch_size)
    global data_input = Tuple(convert_to_dev(dev, data_input[i]) for i in 1:number_data_points)
    global data_output = Tuple(convert_to_dev(dev, data_output[i]) for i in 1:number_data_points)


    loss(in, out, ps, st) = norm(Lux.apply(model, in, ps, st)[1] - out)

    ps, st = Lux.setup(dev, Random.default_rng(), model)

    method = GeometricMachineLearning.AdamOptimizer()
    opt = GeometricMachineLearning.Optimizer(dev, method, model)

    training_error_array = zeros(n_steps)

    #training
    @showprogress for iteration in 1:n_steps
        batch = Int.(ceil.(rand(batch_size)*number_data_points))
        dat_in₁ = data_input[batch[1]]
        dat_out₁   = data_output[batch[1]]
        dp = Zygote.gradient(ps -> loss(dat_in₁, dat_out₁, ps, st), ps)[1]
        Base.Threads.@threads for i in 2:batch_size
            dat_in = data_input[batch[i]]
            dat_out = data_input[batch[i]]
            dp = _add(dp, Zygote.gradient(ps -> loss(dat_in, dat_out, ps, st), ps)[1])
        end

        GeometricMachineLearning.optimization_step!(opt, model, ps, dp)

        training_error_array[iteration] = mapreduce(i -> loss(data_input[i], data_output[i], ps, st), +, 1:number_data_points)
    end
    training_error_array, ps, st
end

n_steps = 10
batch_size = 10000

training_error_cpu, ps_cpu, st = train(CPUDevice(), n_steps, batch_size)

training_error_gpu, ps_gpu, st = train(CUDA.device(), n_steps, batch_size)
