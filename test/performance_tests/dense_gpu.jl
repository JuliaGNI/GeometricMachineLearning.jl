using Lux, Plots, LinearAlgebra, ProgressMeter, CUDA
using GeometricMachineLearning, Random, Zygote

number_data_points = 1000
layer_width = 20
model = Chain(Dense(1, layer_width, tanh), Dense(layer_width, 1))

function train(dev::Device, n_steps, batch_size)
    data_input = 2*π*(1:number_data_points)/number_data_points
    data_output = sin.(data_input)
    #data_input = convert_to_dev(dev, data_input)
    #data_output = convert_to_dev(dev, data_output)

    loss(in, out, ps, st) = norm(Lux.apply(model, in, ps, st)[1] .- out)
    function loss(batch, ps, st)
        mapreduce(i -> loss(convert_to_dev(dev, [data_input[i]]), data_output[i], ps, st), +, batch)
    end

    ps, st = Lux.setup(dev, Random.default_rng(), model)

    method = GeometricMachineLearning.AdamOptimizer()
    opt = GeometricMachineLearning.Optimizer(dev, method, model)

    training_error_array = zeros(n_steps)

    #training
    @showprogress for iteration in 1:n_steps
        batch = Int.(ceil.(rand(batch_size)*number_data_points))

        dp = Zygote.gradient(ps -> loss(batch, ps, st), ps)[1]

        GeometricMachineLearning.optimization_step!(opt, model, ps, dp)

        training_error_array[iteration] = loss(1:number_data_points, ps, st)
    end
    training_error_array, ps, st
end

n_steps = 10000
batch_size = 10

training_error_cpu, ps_cpu, st = train(CPUDevice(), n_steps, batch_size)

training_error_gpu, ps_gpu, st = train(CUDA.device(), n_steps, batch_size)
