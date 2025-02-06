using GeometricMachineLearning

using GeometricMachineLearning

act1 = GeometricMachineLearning.VectorSoftmax()
act2 = GeometricMachineLearning.MatrixSoftmax()

using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate
using LaTeXStrings
using CairoMakie
CairoMakie.activate!()
import Random
Random.seed!(123)

const tstep = .3
const n_init_con = 5

# ensemble problem
ep = hodeensemble([rand(2) for _ in 1:n_init_con], [rand(2) for _ in 1:n_init_con]; tstep = tstep)
dl = DataLoader(integrate(ep, ImplicitMidpoint()); suppress_info = true)


const seq_length = 4
const batch_size = 1024
const n_epochs = 1000

arch1 = StandardTransformerIntegrator(dl.input_dim; transformer_dim = 20,
                                                    n_heads = 4, 
                                                    L = 1, 
                                                    n_blocks = 2,
                                                    attention_activation = act1)

arch2 = StandardTransformerIntegrator(dl.input_dim; transformer_dim = 20,
                                                    n_heads = 4,
                                                    L = 1,
                                                    n_blocks = 2,
                                                    attention_activation = act2)

nn1 = NeuralNetwork(arch1)
nn2 = NeuralNetwork(arch2)

o_method = AdamOptimizer()

o1 = Optimizer(o_method, nn1)
o2 = Optimizer(o_method, nn2)

batch = Batch(batch_size, seq_length)

loss_array1 = o1(nn1, dl, batch, n_epochs)
loss_array2 = o2(nn2, dl, batch, n_epochs)

function make_training_error_plot(n_steps = n_steps; theme = :dark)
    textcolor = theme == :dark ? :white : :black
    fig = Figure(; backgroundcolor = :transparent)
    ax = Axis(fig[1, 1]; 
        backgroundcolor = :transparent,
        bottomspinecolor = textcolor, 
        topspinecolor = textcolor,
        leftspinecolor = textcolor,
        rightspinecolor = textcolor,
        xtickcolor = textcolor, 
        ytickcolor = textcolor,
        xticklabelcolor = textcolor,
        yticklabelcolor = textcolor,
        xlabel=L"t", 
        ylabel=L"q_1",
        xlabelcolor = textcolor,
        ylabelcolor = textcolor,
    )

    # we use linewidth  = 2
    lines!(ax, loss_array1; color = mpurple, label = "VecSoftM", linewidth = 2)
    lines!(ax, loss_array2; color = mred, label = "MatSoftM", linewidth = 2)
    axislegend(; position = (.55, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

training_fig_light, training_ax_light = make_training_error_plot(n_steps; theme = :light)
training_fig_dark, training_ax_dark = make_training_error_plot(n_steps; theme = :dark)

const index = 1
init_con = (q = dl.input.q[:, 1:seq_length, index], p = dl.input.p[:, 1:seq_length, index])

const n_steps = 300

function make_validation_plot(n_steps = n_steps; theme = :dark)
    textcolor = theme == :dark ? :white : :black
    fig = Figure(; backgroundcolor = :transparent)
    ax = Axis(fig[1, 1]; 
        backgroundcolor = :transparent,
        bottomspinecolor = textcolor, 
        topspinecolor = textcolor,
        leftspinecolor = textcolor,
        rightspinecolor = textcolor,
        xtickcolor = textcolor, 
        ytickcolor = textcolor,
        xticklabelcolor = textcolor,
        yticklabelcolor = textcolor,
        xlabel=L"t", 
        ylabel=L"q_1",
        xlabelcolor = textcolor,
        ylabelcolor = textcolor,
    )
    prediction_vector = iterate(nn1, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_matrix = iterate(nn2, init_con; n_points = n_steps, prediction_window = seq_length)

    # we use linewidth  = 2
    lines!(ax, dl.input.q[1, 1:n_steps, index]; color = mblue, label = "Implicit midpoint", linewidth = 2)
    lines!(ax, prediction_vector.q[1, :]; color = mpurple, label = "VecSoftM", linewidth = 2)
    lines!(ax, prediction_matrix.q[1, :]; color = mred, label = "MatSoftM", linewidth = 2)
    axislegend(; position = (.55, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

fig_light, ax_light = make_validation_plot(n_steps; theme = :light)
fig_dark, ax_dark = make_validation_plot(n_steps; theme = :dark)