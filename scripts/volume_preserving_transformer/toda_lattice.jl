using GeometricMachineLearning
using GeometricMachineLearning: transformer_loss, map_to_cpu
using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.TodaLattice: hodeproblem, default_parameters, tspan, hamiltonian, Ñ, p̃₀, q̃₀
using GeometricEquations: EnsembleProblem
using LinearAlgebra: norm 
using Zygote: gradient
import Random 
using CUDA

Random.seed!(123)

const tstep = 1.3

const attention_only = false

# hyperparameters for the problem 
params_collection = [(N = Ñ, α = α̃) for α̃ in [.8]]

ensemble_problem = EnsembleProblem(hodeproblem().equation, tspan, tstep, (q = q̃₀, p = p̃₀), params_collection)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl_nt = DataLoader(ensemble_solution)

# hyperparameters concerning architecture 
const sys_dim = size(dl_nt.input.q, 1) * 2
const n_heads = 2
const L = 1 # transformer blocks 
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
const n_epochs = 200
const batch_size = 1024
const seq_length = 5
const opt_method = AdamOptimizer(T)
const resnet_activation = tanh

# parameters for evaluation 
const α_eval = 0.8 
params = (N = Ñ, α = α_eval)
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
nn₂, loss_array₂ = !attention_only ? setup_and_train(model₂, feedforward_batch, transformer=false) : (nothing, nothing)
nn₃, loss_array₃ = !attention_only ? setup_and_train(model₃, transformer_batch, transformer=true) : (nothing, nothing)

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

nn₁_solution = iterate(nn₁, numerical[:, 1:seq_length]; n_points = length(t_array))
nn₂_solution = !attention_only ? iterate(nn₂, numerical[:, 1]; n_points = length(t_array)) : nothing
nn₃_solution = !attention_only ? iterate(nn₃, numerical[:, 1:seq_length]; n_points = length(t_array)) : nothing

########################### plot validation

p_validation = plot(t_array, numerical[1, :], label = "numerical solution", color = 1, linewidth = 2)

plot!(p_validation, t_array, nn₁_solution[1, :], label = "attention only", color = 2, linewidth = 2)

if !attention_only
    plot!(p_validation, t_array, nn₂_solution[1, :], label = "feedforward", color = 3, linewidth = 2)
    plot!(p_validation, t_array, nn₃_solution[1, :], label = "transformer", color = 4, linewidth = 2)
end

########################### plot training loss

p_training_loss = plot(loss_array₁, label = "attention only", color = 2, linewidth = 2)

if !attention_only
    plot!(p_training_loss, loss_array₂, label = "feedforward", color = 3, linewidth = 2)
    plot!(p_training_loss, loss_array₃, label = "transformer", color = 4, linewidth = 2)
end

png(p_validation, joinpath(@__DIR__, "toda_lattice/validation"))
png(p_training_loss, joinpath(@__DIR__, "toda_lattice/training_loss"))