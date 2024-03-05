using GeometricMachineLearning
using GeometricMachineLearning: transformer_loss, apply_toNT, map_to_cpu
# using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.LotkaVolterra3d: lotka_volterra_3d_ode, default_parameters, tspan, Δt, hamiltonian, X₀, Y₀, Z₀
using GeometricEquations: EnsembleProblem
using Zygote: gradient
using LinearAlgebra: norm

# hyperparameters for the problem 
const q₀ = (q = [X₀, Y₀, Z₀], )

const parameter_ensemble = [(A1 = A₁, A2 = A₁, A3 = 1., B1 = 0., B2 = 1., B3 = 1.) for A₁ in .1 : .1 : 1.]

ensemble_problem = EnsembleProblem(lotka_volterra_3d_ode().equation, tspan, Δt, q₀, parameter_ensemble)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

const dl = DataLoader(ensemble_solution)

# hyperparameters concerning architecture 
const sys_dim = size(dl.input, 1) 
const n_heads = 3
const L = 1 # transformer blocks 
const activation = tanh
const n_linear = 1
const n_blocks = 1
const skew_sym = false

# type and backend 
const backend = CPU()
const T = eltype(dl)

# hyperparameters concerning training 
const n_epochs = 200
const batch_size = 1024
const seq_length = 4
const opt_method = AdamOptimizer(T(0.01))


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

model₁ = Chain(VolumePreservingAttention(sys_dim, seq_length, skew_sym=skew_sym))

# only two linear layers
model₂ = VolumePreservingFeedForward(sys_dim, L * n_blocks, n_linear, activation)

model₃ = VolumePreservingTransformer(sys_dim, seq_length, n_blocks, n_linear, L, activation, skew_sym=skew_sym)

model₄ = RegularTransformerIntegrator(sys_dim, sys_dim, n_heads, L)


"""
This is necessary for now when using Enzyme. 
"""
function dummy_setup()
    dummy_model₁ = VolumePreservingAttention(sys_dim, seq_length, skew_sym=skew_sym)
    ps₁ = initialparameters(backend, T, dummy_model₁)

    dummy_model₂ = Chain(VolumePreservingLowerLayer(sys_dim), VolumePreservingUpperLayer(sys_dim))
    ps₂ = initialparameters(backend, T, dummy_model₂)

    gradient(ps -> norm(dummy_model₁(rand(sys_dim, seq_length, 1), ps)), ps₁)
    gradient(ps -> norm(dummy_model₂(rand(sys_dim, 1, 1), ps)), ps₂)

    nothing
end

dummy_setup()

# nn₁, loss_array₁ = setup_and_train(model₁, transformer_batch, transformer=true)
nn₂, loss_array₂ = setup_and_train(model₂, feedforward_batch, transformer=false)
nn₃, loss_array₃ = setup_and_train(model₃, transformer_batch, transformer=true)
nn₄, loss_array₄ = setup_and_train(model₄, transformer_batch, transformer=true)