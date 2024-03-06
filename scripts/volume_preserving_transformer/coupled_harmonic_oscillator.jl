using GeometricMachineLearning
using GeometricMachineLearning: transformer_loss, map_to_cpu
using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.CoupledHarmonicOscillator: hodeproblem, default_parameters, tspan, tstep, hamiltonian, p₀, q₀
using GeometricEquations: EnsembleProblem
using LinearAlgebra: norm 
using Zygote: gradient
import Random 
using CUDA

Random.seed!(123)

# hyperparameters for the problem 
const m₁ = default_parameters.m₁
const m₂ = default_parameters.m₂
const k₁ = default_parameters.k₁
const k₂ = default_parameters.k₂
const k = Float64.(0.0:0.1:4)

params_collection = [(m₁ = m₁, m₂ = m₂, k₁ = k₁, k₂ = k₂, k = k_val) for k_val in k]
ensemble_problem = EnsembleProblem(hodeproblem().equation, tspan, tstep, (q = q₀, p = p₀), params_collection)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl_nt = DataLoader(ensemble_solution)

# hyperparameters concerning architecture 
const sys_dim = size(dl_nt.input.q, 1) * 2
const n_heads = 2
const L = 2 # transformer blocks 
const activation = tanh
const n_linear = 2
const n_blocks = 3
const skew_sym = false

# backend 
const backend = CUDABackend()

# data loader 
const dl = backend == CPU() ? DataLoader(vcat(dl_nt.input.q, dl_nt.input.p)) : DataLoader(vcat(dl_nt.input.q, dl_nt.input.p) |> cu)

const T = eltype(dl)

# hyperparameters concerning training 
const n_epochs = 20000
const batch_size = 1024
const seq_length = 5
const opt_method = AdamOptimizer(T)
const resnet_activation = tanh

# parameters for evaluation 
const k_eval = 3.5 
params = (m₁ = m₁, m₂ = m₂, k₁ = k₁, k₂ = k₂, k = k_eval)
const t_validation = 30

# model₂ = RegularTransformerIntegrator(sys_dim, transformer_dim, n_heads, L, upscaling_activation, resnet_activation)
# model₃ = VolumePreservingTransformer(sys_dim, seq_length, depth, transformer_dim, L, resnet_activation)

function setup_and_train(model::Union{GeometricMachineLearning.Architecture, GeometricMachineLearning.Chain}, batch::Batch; transformer::Bool=true)
    nn₀ = NeuralNetwork(model, backend, T)
    o₀ = Optimizer(opt_method, nn₀)

    loss_array = transformer ? o₀(nn₀, dl, batch, n_epochs, transformer_loss) : o₀(nn₀, dl, batch, n_epochs)

    GeometricMachineLearning.map_to_cpu(nn₀), loss_array
end

feedforward_batch = Batch(batch_size, 1)
transformer_batch = Batch(batch_size, seq_length)

# attention only
model₁ = Chain(VolumePreservingAttention(sys_dim, seq_length; skew_sym = skew_sym))

model₂ = VolumePreservingFeedForward(sys_dim, n_blocks * L, n_linear, resnet_activation)

# model₂ = RegularTransformerIntegrator(sys_dim, transformer_dim, n_heads, L, upscaling_activation, resnet_activation)
model₃ = VolumePreservingTransformer(sys_dim, seq_length, n_blocks, n_linear, L, resnet_activation; skew_sym = skew_sym)

nn₁, loss_array₁ = setup_and_train(model₁, transformer_batch, transformer=true)
nn₂, loss_array₂ = setup_and_train(model₂, feedforward_batch, transformer=false)
nn₃, loss_array₃ = setup_and_train(model₃, transformer_batch, transformer=true)

function numerical_solution(sys_dim::Int, t_integration::Int, tstep::Real, params::NamedTuple)
    validation_problem = hodeproblem(; tspan = (0.0, t_integration), tstep = tstep, params = params)
    sol = integrate(validation_problem, ImplicitMidpoint())

    numerical_solution = zeros(sys_dim, length(sol.t))
    for i in axes(sol.t, 1) numerical_solution[1:(sys_dim ÷ 2), i+1] = sol.q[i] end 
    for i in axes(sol.t, 1) numerical_solution[(sys_dim ÷ 2 + 1):sys_dim, i+1] = sol.p[i] end

    t_array = zeros(length(sol.t))
    for i in axes(sol.t, 1) t_array[i+1] = sol.t[i] end

    T.(numerical_solution), T.(t_array) 
end

numerical, t_array = numerical_solution(sys_dim, t_validation, tstep, params)

struct DummyTransformer <: GeometricMachineLearning.TransformerIntegrator 
    seq_length::Int
end
nn₁ = NeuralNetwork(DummyTransformer(seq_length), nn₁.model, nn₁.params)

nn₁_solution = iterate(nn₁, numerical[:, 1:seq_length]; n_points = Int(t_validation / tstep) + 1)
nn₂_solution = iterate(nn₂, numerical[:, 1]; n_points = Int(t_validation / tstep) + 1)
nn₃_solution = iterate(nn₃, numerical[:, 1:seq_length]; n_points = Int(t_validation / tstep) + 1)

########################### plot validation

p_validation = plot(t_array, numerical[1, :], label = "numerical solution", color = 1, linewidth = 2)

plot!(p_validation, t_array, nn₁_solution[1, :], label = "attention only", color = 2, linewidth = 2)

plot!(p_validation, t_array, nn₂_solution[1, :], label = "feedforward", color = 3, linewidth = 2)

plot!(p_validation, t_array, nn₃_solution[1, :], label = "transformer", color = 4, linewidth = 2)

########################### plot training loss

p_training_loss = plot(loss_array₁, label = "attention only", color = 2, linewidth = 2)

plot!(p_training_loss, loss_array₂, label = "feedforward", color = 3, linewidth = 2)

plot!(p_training_loss, loss_array₃, label = "transformer", color = 4, linewidth = 2)

png(p_validation, joinpath(@__DIR__, "coupled_harmonic_oscillator/validation"))
png(p_training_loss, joinpath(@__DIR__, "coupled_harmonic_oscillator/training_loss"))