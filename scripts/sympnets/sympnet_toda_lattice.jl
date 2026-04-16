using GeometricMachineLearning
using GeometricMachineLearning: MatrixSoftmax, VectorSoftmax
using GeometricProblems.TodaLattice: hodeproblem, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate
using LaTeXStrings
using CairoMakie
CairoMakie.activate!()
import Random
Random.seed!(123)

morange = RGBf(255 / 256, 127 / 256, 14 / 256)
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)

# ensemble problem
ep = hodeproblem()
dl = DataLoader(integrate(ep, ImplicitMidpoint()); suppress_info = true)

arch = GSympNet(dl; upscaling_dimension = 4)

nn = NeuralNetwork(arch)
o_method = AdamOptimizer()

o = Optimizer(o_method, nn)

batch = Batch(256)
const n_epochs = 1024

loss_array = o(nn, dl, batch, n_epochs)

function make_training_error_plot(; theme = :dark)
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
    lines!(ax, loss_array; color = mpurple, label = "SympNet", linewidth = 2)
    axislegend(; position = (.55, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

training_fig_light, training_ax_light = make_training_error_plot(; theme = :light)
training_fig_dark, training_ax_dark = make_training_error_plot(; theme = :dark)

const index = 1
init_con = (q = dl.input.q[:, 1], p = dl.input.p[:, 1])

const n_steps = 1200

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
    prediction = iterate(nn, init_con; n_points = n_steps)

    # we use linewidth  = 2
    lines!(ax, dl.input.q[:, n_steps]; color = mblue, label = "Implicit midpoint", linewidth = 2)
    lines!(ax, prediction.q[:, n_steps]; color = mpurple, label = "SympNet", linewidth = 2)
    axislegend(; position = (.55, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

fig_light, ax_light = make_validation_plot(n_steps; theme = :light)
fig_dark, ax_dark = make_validation_plot(n_steps; theme = :dark)

save("SympNet-TodaLattice.png", fig_light)