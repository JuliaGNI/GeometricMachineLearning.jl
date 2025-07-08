using GeometricMachineLearning
using GeometricMachineLearning: MatrixSoftmax, VectorSoftmax
using GeometricProblems.DoublePendulum: timespan, timestep, default_parameters, hodeproblem
using GeometricEquations: EnsembleProblem
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
initial_conditions = [
    (q = [π / i, π / j], p = [0.0, π / k]) for i=1:1:10, j=1:1:10, k=1:1:5
]
initial_conditions = reshape(initial_conditions, length(initial_conditions))

ensemble_problem = EnsembleProblem(hodeproblem().equation, (timespan[1], timespan[1] + 15 * timestep), timestep, initial_conditions, default_parameters)

ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl = DataLoader(ensemble_solution)


const seq_length = 5
const transformer_dim = 4

act = MatrixSoftmax()

arch1 = StandardTransformerIntegrator(dl.input_dim; transformer_dim = transformer_dim,
                                                    n_heads = 1, 
                                                    L = 1, 
                                                    n_blocks = 4,
                                                    attention_activation = act)

arch2 = SymplecticTransformer(dl.input_dim; transformer_dim = transformer_dim,
                                            L = 1,
                                            n_sympnet = 6,
                                            attention_activation = act,
                                            symmetric = true)

arch3 = SymplecticTransformer(dl.input_dim; transformer_dim = transformer_dim,
                                            L =1,
                                            n_sympnet = 6,
                                            attention_activation = act,
                                            symmetric = false)


arch4 = GSympNet(dl.input_dim; n_layers = 6)

nn1 = NeuralNetwork(arch1)
nn2 = NeuralNetwork(arch2)
nn3 = NeuralNetwork(arch3)
nn4 = NeuralNetwork(arch4)

const batch_size = 1024
const n_epochs = 3000

o_method = AdamOptimizer()

o1 = Optimizer(o_method, nn1)
o2 = Optimizer(o_method, nn2)
o3 = Optimizer(o_method, nn3)
o4 = Optimizer(o_method, nn4)

batch = Batch(batch_size, seq_length)

loss_array1 = o1(nn1, dl, batch, n_epochs)
loss_array2 = o2(nn2, dl, batch, n_epochs)
loss_array3 = o3(nn3, dl, batch, n_epochs)
loss_array4 = o4(nn4, dl, Batch(batch_size), n_epochs)

function make_training_error_plot(; theme = :dark, symplectic = true)
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
    lines!(ax, loss_array1; color = mpurple, label = "Transformer", linewidth = 2)
    symplectic ? lines!(ax, loss_array2; color = mred, label = "SymplecticTransformerA", linewidth = 2) : nothing
    symplectic ? lines!(ax, loss_array3; color = mgreen, label = "SymplecticTransformerA", linewidth = 2) : nothing
    axislegend(; position = (.55, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

training_fig_light, training_ax_light = make_training_error_plot(; theme = :light)
training_fig_dark, training_ax_dark = make_training_error_plot(; theme = :dark)


function make_plot_for_index(index::Integer=1)
    const n_steps = 100
    ensemble_problem = EnsembleProblem(hodeproblem().equation, (timespan[1], timespan[1] + n_steps * timestep), timestep, [(q = dl.input.q[:, 1, index], p = dl.input.p[:, 1, index]), ], default_parameters)
    ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())
    dl = DataLoader(ensemble_solution)
    init_con = (q = dl.input.q[:, 1:seq_length, 1], p = dl.input.p[:, 1:seq_length, 1])

    # const number_time_steps = 90
    function make_validation_plot(n_steps = n_steps; theme = :dark, symplectic = true)
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
        prediction1 = iterate(nn1, init_con; n_points = n_steps, prediction_window = seq_length)
        prediction2 = iterate(nn2, init_con; n_points = n_steps, prediction_window = seq_length)
        prediction3 = iterate(nn3, init_con; n_points = n_steps, prediction_window = seq_length)

        # we use linewidth  = 2
        lines!(ax, 1:n_steps, dl.input.q[1, 1:n_steps, 1]; color = mblue, label = "Implicit midpoint", linewidth = 2)
        lines!(ax, 1:n_steps, prediction1.q[1, 1:n_steps]; color = mpurple, label = "Transformer", linewidth = 2)
        symplectic ? lines!(ax, 1:n_steps, prediction2.q[1, 1:n_steps]; color = mred, label = "SymplecticTransformerS", linewidth = 2) : false
        symplectic ? lines!(ax, 1:n_steps, prediction3.q[1, 1:n_steps]; color = mgreen, label = "SymplecticTransformerA", linewidth = 2) : false
        axislegend(; position = (.55, .75), backgroundcolor = :transparent, labelcolor = textcolor)

        fig, ax
    end

    fig_light, ax_light = make_validation_plot(n_steps; theme = :light)
    fig_dark, ax_dark = make_validation_plot(n_steps; theme = :dark)

    save("DoublePendulum-Validation_$(index).png", fig_light)
end

make_plot_for_index()