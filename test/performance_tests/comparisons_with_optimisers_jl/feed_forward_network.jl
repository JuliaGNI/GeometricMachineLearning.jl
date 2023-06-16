using Lux, Plots, LinearAlgebra, ProgressMeter
import GeometricMachineLearning, Optimisers, Random, Zygote

number_data_points = 1000

data_input = 2*π*(1:number_data_points)/number_data_points
data_output = sin.(data_input)

loss(in, out, ps, st) = norm(Lux.apply(model, in, ps, st)[1] .- out)
loss(batch, ps, st) = mapreduce(i -> loss([data_input[i]], data_output[i], ps, st), +, batch)

layer_width = 20

model = Chain(Dense(1, layer_width, tanh), Dense(layer_width, 1))

function train_optimisers(n_steps, batch_size)
    ps, st = Lux.setup(Random.default_rng(), model)

    cache_opt = Optimisers.setup(Optimisers.Adam(0.001), ps)

    training_error_array = zeros(n_steps)

    #training
    @showprogress for iteration in 1:n_steps
        batch = Int.(ceil.(rand(batch_size)*number_data_points))

        dp = Zygote.gradient(ps -> loss(batch, ps, st), ps)[1]

        st_opt, ps = Optimisers.update(cache_opt, ps, dp)

        training_error_array[iteration] = loss(1:number_data_points, ps, st)
    end
    training_error_array, ps, st
end

function train_geometric_machine_learning(n_steps, batch_size)
    ps, st = Lux.setup(Random.default_rng(), model)

    method = GeometricMachineLearning.AdamOptimizer()
    opt = GeometricMachineLearning.Optimizer(method, model)

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

training_error_optimisers, ps_optimisers, st = train_optimisers(n_steps, batch_size)
training_error_geometric_machine_learning, ps_geometric_machine_learning, st = train_geometric_machine_learning(n_steps, batch_size)

p = plot(training_error_optimisers, label="Optimisers.jl")
plot!(p, training_error_geometric_machine_learning, label="GeometricMachineLearning.jl")

png(p, "FeedForwardNeuralNetworkTraining")

p₂ = plot(data_input, data_output, label="Training data", colour=3)

output_optimisers = [Lux.apply(model, [data_input[i]], ps_optimisers, st)[1][1] for i in 1:number_data_points]
output_geometric_machine_learning = [Lux.apply(model, [data_input[i]], ps_optimisers, st)[1][1] for i in 1:number_data_points]

plot!(p₂, data_input, output_optimisers, label="Optimisers", colour=1)
plot!(p₂, data_input, output_geometric_machine_learning, label="GeometricMachineLearning", colour=2)

png(p₂, "FeedForwardNeuralNetworkResult")