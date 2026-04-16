using GeometricMachineLearning
using GeometricMachineLearning: map_to_cpu
using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.ABCFlow: odeproblem, default_parameters 
using GeometricEquations: EnsembleProblem
using LinearAlgebra: norm 
using Zygote: gradient
using Metal
import Random 

Random.seed!(123)
const ics₁ = [(q = [0., 0., z_val], ) for z_val in 0.:.01:.9]
const ics₂ = [(q = [0., y_val, 0.], ) for y_val in 0.:.01:.9]
const ics = [ics₁..., ics₂...]

const timestep = .8
const timespan = (0., 1000.)

const sys_dim = length(ics[1].q)

const n_blocks = 6
const n_linear = 2
const activation = tanh

const batch_size = 16384
const opt_method = AdamOptimizer()
const n_epochs = 1000

const backend = MetalBackend()
const T = backend == CPU() ? Float64 : Float32

const t_validation = 5

ensemble_problem = EnsembleProblem(odeproblem().equation, timespan, timestep, ics, default_parameters)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())
dl₁ = DataLoader(ensemble_solution)
dl = backend == CPU() ? dl₁ : DataLoader(dl₁.input |> MtlArray{T})

model = VolumePreservingFeedForward(sys_dim, n_blocks, n_linear, activation)

nn = NeuralNetwork(model, backend, T)

o = Optimizer(opt_method, nn)

batch = Batch(batch_size, 1)

loss_array₁ =  o(nn, dl, batch, n_epochs)

ic = (q = [0., 0., .1], )

function numerical_solution(sys_dim::Int, t_integration::Int, timestep::Real, ic::NamedTuple)
    validation_problem = odeproblem(ic ; timespan = (0.0, t_integration), timestep = timestep, parameters = default_parameters)
    sol = integrate(validation_problem, ImplicitMidpoint())

    numerical_solution = zeros(sys_dim, length(sol.t))
    for i in axes(sol.t, 1) numerical_solution[:, i+1] = sol.q[i] end 

    t_array = zeros(length(sol.t))
    for i in axes(sol.t, 1) t_array[i+1] = sol.t[i] end

    T.(numerical_solution), T.(t_array) 
end

function make_validation_plot(t_validation::Int, nn::NeuralNetwork)

    numerical, t_array = numerical_solution(sys_dim, t_validation, timestep, ic)

    nn₁_solution = iterate(nn₁, numerical[:, 1]; n_points = Int(floor(t_validation / timestep)) + 1)

    ########################### plot validation

    p_validation = plot(t_array, numerical[1, :], label = "numerical solution", color = 1, linewidth = 2)

    plot!(p_validation, t_array, nn₁_solution[1, :], label = "volume-preserving feedforward", color = 2, linewidth = 2)

    p_validation
end

nn₁ = NeuralNetwork(GeometricMachineLearning.DummyNNIntegrator(), nn.model, map_to_cpu(nn.params))

p_validation = make_validation_plot(t_validation, nn₁)

p_validation₂ = make_validation_plot(20, nn₁)
########################### plot training loss

p_training_loss = plot(loss_array₁, label = "volume-preserving feedforward", color = 2, linewidth = 2, yaxis = :log)

png(p_validation, joinpath(@__DIR__, "abc_flow/validation"))
png(p_validation₂, joinpath(@__DIR__, "abc_flow/validation2"))
png(p_training_loss, joinpath(@__DIR__, "abc_flow/training_loss"))