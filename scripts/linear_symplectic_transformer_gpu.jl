using CUDA
using GeometricMachineLearning
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate 
using LaTeXStrings
using CairoMakie
using LinearAlgebra: norm

const timestep = .3
const n_init_con = 1000

# ensemble problem
ep = hodeensemble([rand(2) for _ in 1:n_init_con], [rand(2) for _ in 1:n_init_con]; timestep = timestep)

dl_nt = DataLoader(integrate(ep, ImplicitMidpoint()))
dl = DataLoader(vcat(dl_nt.input.q, dl_nt.input.p) |> cu)

const seq_length = 4
const batch_size = 16384
const n_epochs = 2000

arch_standard = StandardTransformerIntegrator(dl.input_dim; n_heads = 2)
arch_symplectic = LinearSymplecticTransformer(dl.input_dim, seq_length; n_sympnet = 2, L = 1, upscaling_dimension = 5 * dl.input_dim)
arch_sympnet = GSympNet(dl.input_dim; n_layers = 4, upscaling_dimension = 5 * dl.input_dim)

nn_standard = NeuralNetwork(arch_standard, CUDABackend())
nn_symplectic = NeuralNetwork(arch_symplectic, CUDABackend())
nn_sympnet = NeuralNetwork(arch_sympnet, CUDABackend())

o_method = AdamOptimizer()

o_standard = Optimizer(o_method, nn_standard)
o_symplectic = Optimizer(o_method, nn_symplectic)
o_sympnet = Optimizer(o_method, nn_sympnet)

batch = Batch(batch_size, seq_length)
batch2 = Batch(batch_size)

loss_array_standard = o_standard(nn_standard, dl, batch, n_epochs)
loss_array_symplectic = o_symplectic(nn_symplectic, dl, batch, n_epochs)
loss_array_sympnet = o_sympnet(nn_sympnet, dl, batch2, n_epochs)

fig_train = Figure()
ax_train = Axis(fig_train[1, 1]; xlabel = "epoch", ylabel = "training error", yscale = log10)
lines!(ax_train, loss_array_standard; color = Makie.wong_colors()[2], label = "ST")
lines!(ax_train, loss_array_symplectic; color = Makie.wong_colors()[4], label = "LST")
lines!(ax_train, loss_array_sympnet; color = Makie.wong_colors()[3], label = "SympNet")
axislegend(ax_train)

function _convert_to_cpu(dl, nn_standard, nn_symplectic, nn_sympnet)
    DataLoader(dl.input |> Array{Float32}), GeometricMachineLearning.map_to_cpu(nn_standard), GeometricMachineLearning.map_to_cpu(nn_symplectic), GeometricMachineLearning.map_to_cpu(nn_sympnet)
end

dl, nn_standard, nn_symplectic, nn_sympnet =  _convert_to_cpu(dl, nn_standard, nn_symplectic, nn_sympnet)

const index = 1
init_con = dl.input[:, 1:seq_length, index]

const n_steps = 50

function make_validation_plot(n_steps = n_steps; kwargs...)
    prediction_implicit_midpoint = dl.input[:, 1:n_steps, index]
    prediction_standard = iterate(nn_standard, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_symplectic = iterate(nn_symplectic, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_sympnet = iterate(nn_sympnet, init_con[:, 1]; n_points = n_steps)

    fig_validate = Figure()
    ax_validate = Axis(fig_validate[1, 1]; ylabel = L"q_1", kwargs...)
    lines!(ax_validate, prediction_implicit_midpoint[1, :]; color = Makie.wong_colors()[1], label = "implicit midpoint")
    lines!(ax_validate, prediction_standard[1, :]; color = Makie.wong_colors()[2], label = "ST")
    lines!(ax_validate, prediction_symplectic[1, :]; color = Makie.wong_colors()[4], label = "LST")
    lines!(ax_validate, prediction_sympnet[1, :]; color = Makie.wong_colors()[3], label = "SympNet")
    axislegend(ax_validate)
    p_validate = fig_validate

    p_validate, norm(prediction_implicit_midpoint - prediction_standard), norm(prediction_implicit_midpoint - prediction_symplectic), norm(prediction_implicit_midpoint - prediction_sympnet)
end

p_validate, error_standard, error_symplectic, error_sympnet = make_validation_plot()