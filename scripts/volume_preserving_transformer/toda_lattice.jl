using GeometricMachineLearning
using GeometricMachineLearning: transformer_loss, map_to_cpu
# using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.TodaLattice: hodeproblem, default_parameters, tspan, tstep, hamiltonian, Ñ, p̃₀, q̃₀
using GeometricEquations: EnsembleProblem
using LinearAlgebra: norm 
using Zygote: gradient
import Random 

Random.seed!(123)

param_collection = [(N = Ñ, α = α) for α in .5:.1:1.]

ensemble_problem = EnsembleProblem(hodeproblem().equation, tspan, tstep, (q = q̃₀, p = p̃₀), param_collection)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl_nt = DataLoader(ensemble_solution)

# hyperparameters concerning architecture 
const sys_dim = size(dl_nt.input.q, 1) * 2
const n_heads = 2
const L = 1 # transformer blocks 
const activation = tanh
const n_linear = 1
const n_blocks = 1

# type and backend 
const backend = CPU()
const T = eltype(dl_nt)

# type and backend 
const backend = CPU()
const T = eltype(dl_nt)

# data loader 
const dl = DataLoader(vcat(dl_nt.input.q, dl_nt.input.p)) 

# hyperparameters concerning training 
const n_epochs = 10
const batch_size = 1024
const seq_length = 5
const opt_method = AdamOptimizer(T)
const resnet_activation = tanh


# model₂ = RegularTransformerIntegrator(sys_dim, transformer_dim, n_heads, L, upscaling_activation, resnet_activation)
# model₃ = VolumePreservingTransformer(sys_dim, seq_length, depth, transformer_dim, L, resnet_activation)

function setup_and_train(model::Union{GeometricMachineLearning.Architecture, GeometricMachineLearning.Chain}, batch::Batch; transformer::Bool=true)
    nn₀ = NeuralNetwork(model, backend, T)
    o₀ = Optimizer(opt_method, nn₀)

    loss_array = transformer ? o₀(nn₀, dl, batch, n_epochs, transformer_loss) : o₀(nn₀, dl, batch, n_epochs)

    nn₀, loss_array
end

feedforward_batch = Batch(batch_size, 1)
transformer_batch = Batch(batch_size, seq_length)

model₁ = Chain(VolumePreservingAttention(sys_dim, seq_length))

# only two linear layers
model₂ = VolumePreservingFeedForward(sys_dim, n_blocks * L, n_linear)

# model₂ = RegularTransformerIntegrator(sys_dim, transformer_dim, n_heads, L, upscaling_activation, resnet_activation)
model₃ = VolumePreservingTransformer(sys_dim, seq_length, n_blocks, n_linear, L, resnet_activation)

nn₁, loss_array₁ = setup_and_train(model₁, transformer_batch, transformer=true)
nn₂, loss_array₂ = setup_and_train(model₂, feedforward_batch, transformer=false)
nn₃, loss_array₃ = setup_and_train(model₃, transformer_batch, transformer=true)