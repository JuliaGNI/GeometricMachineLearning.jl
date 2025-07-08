using Zygote: gradient, pullback
using GeometricMachineLearning
using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.DoublePendulum: hodeproblem, default_parameters, timespan, hamiltonian, ϑ
using GeometricEquations: EnsembleProblem
using LinearAlgebra: norm 
using Zygote: gradient
import Random 

Random.seed!(123)

θ₀ = [[π / 4, π / i] for i in 1:20]
ω₀ = [0.0, π / 8]
p₀ = [ϑ(timespan[begin], θ, ω₀, default_parameters) for θ in θ₀]
const timestep = .06

ensemble_problem = EnsembleProblem(hodeproblem().equation, timespan, timestep, [(q = q, p = p) for (q, p) in zip(θ₀, p₀)], default_parameters)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl_nt = DataLoader(ensemble_solution)

# hyperparameters concerning architecture 
const sys_dim = size(dl_nt.input.q, 1) * 2
const activation = tanh
const n_linear = 2
const n_blocks = 3

# type and backend 
const backend = CPU()

# data loader 
const dl = DataLoader(vcat(dl_nt.input.q, dl_nt.input.p) |> Array{Float32}) 

const T = eltype(dl)

# hyperparameters concerning training 
const n_epochs = 1000
const batch_size = 1024
const opt_method = AdamOptimizer(T)
const resnet_activation = tanh

const t_validation = 30

# model₂ = RegularTransformerIntegrator(sys_dim, transformer_dim, n_heads, L, upscaling_activation, resnet_activation)
# model₃ = VolumePreservingTransformer(sys_dim, seq_length, depth, transformer_dim, L, resnet_activation)

function setup_and_train(model::Union{GeometricMachineLearning.Architecture, GeometricMachineLearning.Chain}, batch::Batch; transformer::Bool=true)
    nn₀ = NeuralNetwork(model, backend, T)
    o₀ = Optimizer(opt_method, nn₀)

    loss_array = transformer ? o₀(nn₀, dl, batch, n_epochs, transformer_loss) : o₀(nn₀, dl, batch, n_epochs)

    nn₀, loss_array
end

feedforward_batch = Batch(batch_size, 1)

model₂ = VolumePreservingFeedForward(sys_dim, n_blocks, n_linear, resnet_activation)

nn₂, loss_array₂ = setup_and_train(model₂, feedforward_batch, transformer=false)

function numerical_solution(ics::NamedTuple, sys_dim::Int, t_integration::Int, timestep::Real, params::NamedTuple)
    validation_problem = hodeproblem(ics.q, ics.p; timespan = (0.0, t_integration), timestep = timestep, params = params)
    sol = integrate(validation_problem, ImplicitMidpoint())

    numerical_solution = zeros(sys_dim, length(sol.t))
    for i in axes(sol.t, 1) numerical_solution[1:(sys_dim ÷ 2), i+1] = sol.q[i] end 
    for i in axes(sol.t, 1) numerical_solution[(sys_dim ÷ 2 + 1):sys_dim, i+1] = sol.p[i] end

    t_array = zeros(length(sol.t))
    for i in axes(sol.t, 1) t_array[i+1] = sol.t[i] end

    T.(numerical_solution), T.(t_array) 
end

θ₀_val = [π / 4, π / 4]
ω₀_val = [0.0, π / 8]
p₀_val = ϑ(timespan[begin], θ₀_val, ω₀, default_parameters) 

ics = (q = θ₀_val, p = p₀_val)
numerical, t_array = numerical_solution(ics, sys_dim, t_validation, timestep, default_parameters)
nn₂_solution = iterate(nn₂, numerical[:, 1]; n_points = Int(t_validation / timestep) + 1)

p_validation = plot(t_array, numerical[1, :], label = "numerical solution", color = 1, linewidth = 2)
plot!(p_validation, t_array, nn₂_solution[1, :], label = "feedforward", color = 3, linewidth = 2)
