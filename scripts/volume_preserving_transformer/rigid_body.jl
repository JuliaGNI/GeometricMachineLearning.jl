using CUDA
using GeometricMachineLearning
using GeometricMachineLearning: map_to_cpu
# using Plots; pyplot()
using JLD2
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.RigidBody: odeproblem, odeensemble, default_parameters
using LaTeXStrings
import Random 

# hyperparameters for the problem 
const timestep = .2
const timespan = (0., 20.)
const ics₁ = [[sin(val), 0., cos(val)] for val in .1:.01:(2*π)]
const ics₂ = [[0., sin(val), cos(val)] for val in .1:.01:(2*π)]
const ics = [ics₁..., ics₂...]

ensemble_problem = odeensemble(ics; timespan = timespan, timestep = timestep, parameters = default_parameters)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl₁ = DataLoader(ensemble_solution)

# hyperparameters concerning architecture 
const sys_dim = size(dl₁.input, 1)
const n_heads = 1
const L = 3 # transformer blocks 
const activation = tanh
const resnet_activation = tanh
const n_linear = 1
const n_blocks = 2
const skew_sym = true
const seq_length = 3

# backend 
const backend = CUDABackend()

# data type 
const T = Float32

# data loader 
const dl = backend == CPU() ? DataLoader(dl₁.input) : DataLoader(dl₁.input |> CuArray{T})

# hyperparameters concerning training 
const n_epochs = 500000
const batch_size = 16384
const opt_method = AdamOptimizerWithDecay(n_epochs, T; η₁ = 1e-2, η₂ = 1e-6)

# parameters for evaluation 
ics_val = [sin(1.1), 0., cos(1.1)]
ics_val₂ = [0., sin(1.1), cos(1.1)]
const t_validation = 14
const t_validation_long = 100

function train_the_network(nn₀::GeometricMachineLearning.NeuralNetwork, batch::Batch)
    Random.seed!(1234)

    o₀ = Optimizer(opt_method, nn₀)

    loss_array = o₀(nn₀, dl, batch, n_epochs)

    GeometricMachineLearning.map_to_cpu(nn₀), loss_array
end

function setup_and_train(model::GeometricMachineLearning.Architecture, batch::Batch)
    Random.seed!(1234)

    nn₀ = NeuralNetwork(model, backend, T)

    train_the_network(nn₀, batch)
end

function setup_and_train(model::GeometricMachineLearning.Chain, batch::Batch)
    Random.seed!(1234)

    nn₀ = NeuralNetwork(model, backend, T)
    nn₀ = NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), nn₀.model, nn₀.params)

    train_the_network(nn₀, batch)
end

feedforward_batch = Batch(batch_size)
transformer_batch = Batch(batch_size, seq_length, seq_length)

# attention only
# model₁ = Chain(VolumePreservingAttention(sys_dim, seq_length; skew_sym = skew_sym))

model₂ = VolumePreservingFeedForward(sys_dim, n_blocks * L, n_linear, resnet_activation)

model₃ = VolumePreservingTransformer(sys_dim, seq_length; n_blocks = n_blocks, n_linear = n_linear, L = L, activation = resnet_activation, skew_sym = skew_sym)

model₄ = StandardTransformerIntegrator(sys_dim; n_heads = n_heads, transformer_dim = sys_dim, n_blocks = n_blocks, L = L, resnet_activation = resnet_activation, add_connection = false)

# nn₁, loss_array₁ = setup_and_train(model₁, transformer_batch)
nn₂, loss_array₂ = setup_and_train(model₂, feedforward_batch)
nn₃, loss_array₃ = setup_and_train(model₃, transformer_batch)
nn₄, loss_array₄ = setup_and_train(model₄, transformer_batch)

save("transformer_rigid_body.jld2",
        "nn2_params", nn₂.params,
        # "nn2_loss_array", loss_array₂,
        "nn3_params", nn₃.params,
        # "nn3_loss_array", loss_array₃,
        "nn4_params", nn₄.params,
        # "nn4_loss_array", loss_array₄
        )

function numerical_solution(sys_dim::Int, t_integration::Int, timestep::Real, ics_val::Vector)
    validation_problem = odeproblem(ics_val; timespan = (0.0, t_integration), timestep = timestep, parameters = default_parameters)
    sol = integrate(validation_problem, ImplicitMidpoint())

    numerical_solution = zeros(sys_dim, length(sol.t))
    for i in axes(sol.t, 1) numerical_solution[:, i+1] = sol.q[i] end 

    t_array = zeros(length(sol.t))
    for i in axes(sol.t, 1) t_array[i+1] = sol.t[i] end

    T.(numerical_solution), T.(t_array) 
end

function plot_validation(t_validation; nn₂=nn₂, nn₃=nn₃, nn₄=nn₄, plot_regular_transformer = false, plot_vp_transformer = false)

    numerical, t_array = numerical_solution(sys_dim, t_validation, timestep, ics_val)

    # nn₁_solution = iterate(nn₁, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / timestep)) + 1)
    nn₂_solution = iterate(nn₂, numerical[:, 1]; n_points = Int(floor(t_validation / timestep)) + 1)
    nn₃_solution = iterate(nn₃, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / timestep)) + 1, prediction_window = seq_length)
    nn₄_solution = iterate(nn₄, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / timestep)) + 1, prediction_window = seq_length)

    ########################### plot validation

    p_validation = plot(t_array, numerical[1, :], label = "implicit midpoint", color = 1, linewidth = 2, dpi = 400, ylabel = "z", xlabel = "time")

    # plot!(p_validation, t_array, nn₁_solution[1, :], label = "attention only", color = 2, linewidth = 2)

    plot!(p_validation, t_array, nn₂_solution[1, :], label = "feedforward", color = 3, linewidth = 2)

    if plot_vp_transformer
        plot!(p_validation, t_array, nn₃_solution[1, :], label = "transformer", color = 4, linewidth = 2)
    end

    if plot_regular_transformer
        plot!(p_validation, t_array, nn₄_solution[1, :], label = "standard transformer", color = 5, linewidth = 2)
    end

    p_validation
end

p_validation = plot_validation(t_validation; plot_regular_transformer = true, plot_vp_transformer = true)
p_validation_long = plot_validation(t_validation_long)

########################### plot training loss

p_training_loss = plot(loss_array₃, label = "transformer", color = 4, linewidth = 2, yaxis = :log, dpi = 400, ylabel = "training loss", xlabel = "epoch")

# plot!(loss_array₁, label = "attention only", color = 2, linewidth = 2)

plot!(p_training_loss, loss_array₂, label = "feedforward", color = 3, linewidth = 2)

plot!(p_training_loss, loss_array₄, label = "standard transformer", color = 5, linewidth = 2)

########################## plot 3d validation 

function sphere(r, C) # r: radius; C: center [cx,cy,cz]
    n = 100
    u = range(-π, π; length = n)
    v = range(0, π; length = n)
    x = C[1] .+ r*cos.(u) * sin.(v)'
    y = C[2] .+ r*sin.(u) * sin.(v)'
    z = C[3] .+ r*ones(n) * cos.(v)'
    return x, y, z
end

function make_validation_plot3d(t_validation::Int, nn::NeuralNetwork)
    numerical, _ = numerical_solution(sys_dim, t_validation, timestep, ics_val)
    numerical₂, _ = numerical_solution(sys_dim, t_validation, timestep, ics_val₂)

    prediction_window = typeof(nn) <: NeuralNetwork{<:GeometricMachineLearning.TransformerIntegrator} ? seq_length : 1

    nn₁_solution = iterate(nn, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / timestep)) + 1, prediction_window = prediction_window)
    nn₁_solution₂ = iterate(nn, numerical₂[:, 1:seq_length]; n_points = Int(floor(t_validation / timestep)) + 1, prediction_window = prediction_window)

    ########################### plot validation

    p_validation = surface(sphere(1., [0., 0., 0.]), alpha = .2, colorbar = false, dpi = 400, xlabel = L"z_1", ylabel = L"z_2", zlabel = L"z_3", xlims = (-1, 1), ylims = (-1, 1), zlims = (-1, 1), aspect_ratio = :equal)
    
    plot!(p_validation, numerical[1, :], numerical[2, :], numerical[3, :], label = "implicit midpoint", color = 1, linewidth = 2, dpi = 400)
    plot!(p_validation, numerical₂[1, :], numerical₂[2, :], numerical₂[3, :], label = nothing, color = 1, linewidth = 2, dpi = 400)

    plot!(p_validation, nn₁_solution[1, :], nn₁_solution[2,:], nn₁_solution[3, :], label = "volume-preserving transformer", color = 4, linewidth = 2)
    plot!(p_validation, nn₁_solution₂[1, :], nn₁_solution₂[2,:], nn₁_solution₂[3, :], label = nothing, color = 4, linewidth = 2)

    p_validation
end

p_validation3d = make_validation_plot3d(t_validation_long, nn₃)

png(p_validation, joinpath(@__DIR__, "rigid_body/validation_"*string(seq_length)))
png(p_training_loss, joinpath(@__DIR__, "rigid_body/training_loss_"*string(seq_length)))
png(p_validation3d, joinpath(@__DIR__, "rigid_body/validation3d_"*string(seq_length)))