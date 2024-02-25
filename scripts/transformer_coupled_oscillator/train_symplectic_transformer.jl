using GeometricMachineLearning
using GeometricMachineLearning: transformer_loss, apply_toNT, map_to_cpu
# using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.CoupledHarmonicOscillator: hodeproblem, default_parameters, tspan, tstep, q₀, p₀, hamiltonian
using GeometricEquations: EnsembleProblem

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
const transformer_dim = sys_dim
const n_heads = 2
const sympnet_upscaling = 3
const L = 1 # transformer blocks 
const upscaling_activation = identity
const resnet_activation = tanh
const depth = 8
const n_blocks = 2
const n_linear = 2

# type and backend 
const backend = CPU()
const T = eltype(dl_nt)

# data loader 
const dl = DataLoader(vcat(dl_nt.input.q, dl_nt.input.p)) 

# hyperparameters concerning training 
const n_epochs = 200
const batch_size = 1024
const seq_length = 4
const opt_method = AdamOptimizer(T)

model₁ = RegularTransformerIntegrator(sys_dim, transformer_dim, n_heads, L, upscaling_activation, resnet_activation)
model₂ = VolumePreservingTransformer(sys_dim, seq_length, n_blocks, n_linear, L, resnet_activation)

model₃ = Chain(  Dense(sys_dim, transformer_dim, upscaling_activation),
              ResNet(transformer_dim, resnet_activation),
              ResNet(transformer_dim, resnet_activation),
              Dense(transformer_dim, sys_dim, identity)
              )

model₄ = Chain( PSDLayer(sys_dim, transformer_dim),
              GradientLayerQ(transformer_dim, sympnet_upscaling * transformer_dim, resnet_activation),
              GradientLayerP(transformer_dim, sympnet_upscaling * transformer_dim, resnet_activation),
              GradientLayerP(transformer_dim, sympnet_upscaling * transformer_dim, resnet_activation),
              GradientLayerQ(transformer_dim, sympnet_upscaling * transformer_dim, resnet_activation),
	          PSDLayer(transformer_dim, sys_dim)
              )

function setup_and_train(model::Union{GeometricMachineLearning.Architecture, GeometricMachineLearning.Chain}, batch::Batch, transformer::Bool=true)
    nn₀ = NeuralNetwork(model, backend, T)
    o₀ = Optimizer(opt_method, nn₀)

    loss_array = transformer ? o₀(nn₀, dl, batch, n_epochs, transformer_loss) : o₀(nn₀, dl, batch, n_epochs)

    nn₀, loss_array
end

batch = Batch(batch_size, seq_length)
sympnet_batch = Batch(batch_size, 1)

nn₁, loss_array₁ = setup_and_train(model₁, batch)
nn₂, loss_array₂ = setup_and_train(model₂, batch)
nn₃, loss_array₃ = setup_and_train(model₃, sympnet_batch, false)
nn₄, loss_array₄ = setup_and_train(model₄, sympnet_batch, false)

struct DummySympNet{AT} <: GeometricMachineLearning.SympNet{AT} end
function DummySympNet(activation = identity)
    DummySympNet{typeof(activation)}()
end

function convert_nn_to_cpu_sympnet(nn::NeuralNetwork)
    nn_cpu = map_to_cpu(nn)
    nn_validation = NeuralNetwork(DummySympNet(), nn_cpu.model, nn_cpu.params)
end
nn₃ = convert_nn_to_cpu_sympnet(nn₃)
nn₄ = convert_nn_to_cpu_sympnet(nn₄)

@doc raw"""
Computes the numerical solution for the problem (for comparative purposes)
"""
function numerical_solution(; t_integration = t_integration, params = (m₁ = 2., m₂ = 1., k₁ = 1.5, k₂ = 0.3, k = 3.5), only_symplectic::Bool = false)
    validation_problem = hodeproblem(; tspan = (0.0, t_integration), tstep = tstep, params = params)
    sol = integrate(validation_problem, ImplicitMidpoint())

    numerical_solution = zeros(sys_dim, length(sol.t))
    for i in axes(sol.t, 1) numerical_solution[1:2, i+1] = sol.q[i] end 
    for i in axes(sol.t, 1) numerical_solution[3:4, i+1] = sol.p[i] end

    t_array = zeros(length(sol.t))
    for i in axes(sol.t, 1) t_array[i+1] = sol.t[i] end

    numerical_solution, t_array 
end

function transformer_out_of_numerical_solution(nn::NeuralNetwork, numerical_solution::AbstractMatrix)
    ics_for_network = numerical_solution[:, 1:seq_length]

    nn_cpu = map_to_cpu(nn)
    nn_validation = NeuralNetwork(DummyTransformer(), nn_cpu.model, nn_cpu.params)

    GeometricMachineLearning.iterate(nn_validation, ics_for_network, n_points = size(numerical_solution, 2), seq_length = seq_length)
end

function sympnet_out_of_numerical_solution(nn::NeuralNetwork, numerical_solution::AbstractMatrix)
    ics_for_network = numerical_solution[:, 1]
    GeometricMachineLearning.iterate(nn, ics_for_network, n_points = size(numerical_solution, 2))
end

function compare_q(t::Vector, numerical_solution::AT, prediction₁::AT, prediction₂::AT, prediction₃::AT, prediction₄::AT; index::Int = 1) where AT
    p1 = plot(t, im[index, :], label="Numerical Integration", size=(1500, 600), xlabel="time", ylabel="q$(index)", linewidth=2)
    plot!(p1, t, prediction₃[index, :], label="ResNet", linewidth=2)
    plot!(p1, t, prediction₄[index, :], label="SympNet", linewidth=2)
    plot!(p1, t, prediction₁[index, :], label="Regular Transformer", linewidth=2)
    plot!(p1, t, prediction₂[index, :], label="Structure-Preserving Transformer", linewidth=2)
    vline!(p1, [(seq_length - 1) * tstep], label="Start of prediction for transformers", color=:red)
    p1
end

function compare_energies(t::Vector, numerical_solution::AT, prediction₁::AT, prediction₂::AT, only_symplectic::Bool=false; params = (m₁ = 2., m₂ = 1., k₁ = 1.5, k₂ = 0.3, k = 3.5)) where AT 
    plot2 = plot(t, get_energy(t, numerical_solution, params = params), label="Numerical Integration", size=(1000, 600))
    only_symplectic ? nothing : plot!(plot2, t, get_energy(t, prediction₁, params = params), label = "Regular Transformer", color=2) 
    plot!(plot2, t, get_energy(t, prediction₂, params = params), label = "Structure-Preserving Transformer", color=3)

    plot2
end

function save_plot(t_final = 15., index = 1)
    im, t = numerical_solution(; t_integration = t_final)
    prediction₁ = transformer_out_of_numerical_solution(nn₁, im)
    prediction₂ = transformer_out_of_numerical_solution(nn₂, im)
    prediction₃ = sympnet_out_of_numerical_solution(nn₃, im)
    prediction₄ = sympnet_out_of_numerical_solution(nn₄, im)

    p3 = compare_q(t, im, prediction₁, prediction₂, prediction₃, prediction₄, index = index)
    savefig(p3, "q$(index)_$(t_final).pdf")
end


display(GeometricMachineLearning.parameterlength(nn₁))
display(GeometricMachineLearning.parameterlength(nn₂))

display(GeometricMachineLearning.parameterlength(nn₃))
display(GeometricMachineLearning.parameterlength(nn₄))