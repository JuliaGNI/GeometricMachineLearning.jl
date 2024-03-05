using Zygote: gradient, pullback
using GeometricMachineLearning
using GeometricMachineLearning: tensor_mat_skew_sym_assign, mat_tensor_mul
# using Plots

#=
"""
This is necessary for now when using Enzyme. 
"""
function dummy_setup(sys_dim = 2, seq_length = 4)

    pullback(tensor_mat_skew_sym_assign, rand(sys_dim, seq_length, 1), rand(sys_dim, sys_dim))[2](rand(sys_dim, seq_length, 1))
    pullback(mat_tensor_mul, rand(LowerTriangular, sys_dim), rand(sys_dim, seq_length, 1))[2](rand(sys_dim, seq_length, 1))
    pullback(mat_tensor_mul, rand(UpperTriangular, sys_dim), rand(sys_dim, seq_length, 1))[2](rand(sys_dim, seq_length, 1))
    pullback(mat_tensor_mul, rand(SkewSymMatrix, sys_dim), rand(sys_dim, seq_length, 1))[2](rand(sys_dim, seq_length, 1))

    nothing
end

dummy_setup(2, 4)
=# 
# const sys_dim = 4
# const seq_length = 4 
# pullback(tensor_mat_skew_sym_assign, rand(sys_dim, seq_length, 1), rand(sys_dim, sys_dim))[2](rand(sys_dim, seq_length, 1))
# pullback(mat_tensor_mul, rand(LowerTriangular, sys_dim), rand(sys_dim, seq_length, 1))[2](rand(sys_dim, seq_length, 1))
# pullback(mat_tensor_mul, rand(UpperTriangular, sys_dim), rand(sys_dim, seq_length, 1))[2](rand(sys_dim, seq_length, 1))
# pullback(mat_tensor_mul, rand(SkewSymMatrix, sys_dim), rand(sys_dim, seq_length, 1))[2](rand(sys_dim, seq_length, 1))

using GeometricMachineLearning
using GeometricMachineLearning: transformer_loss, apply_toNT, map_to_cpu
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.DoublePendulum: hodeproblem, default_parameters, tspan, tstep, hamiltonian, ϑ
using GeometricEquations: EnsembleProblem
using LinearAlgebra: norm 

θ₀ = [[π / 4, π / i] for i in 1:1]
ω₀ = [0.0, π / 8]
p₀ = [ϑ(tspan[begin], θ, ω₀, default_parameters) for θ in θ₀]

ensemble_problem = EnsembleProblem(hodeproblem().equation, tspan, tstep, [(q = q, p = p) for (q, p) in zip(θ₀, p₀)], default_parameters)
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
const n_epochs = 50000
const batch_size = 1024
const seq_length = 4
const opt_method = AdamOptimizer(T)


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
model₂ = VolumePreservingFeedForward(sys_dim, n_blocks, n_linear, activation)

# model₂ = RegularTransformerIntegrator(sys_dim, transformer_dim, n_heads, L, upscaling_activation, resnet_activation)

model₃ = VolumePreservingTransformer(sys_dim, seq_length, n_blocks, n_linear, L, activation)

nn₁, loss_array₁ = setup_and_train(model₁, transformer_batch, transformer=true)
nn₂, loss_array₂ = setup_and_train(model₂, feedforward_batch, transformer=false)
nn₃, loss_array₃ = setup_and_train(model₃, transformer_batch, transformer=true)