# Comparing Matrix and Vector Softmax as Activation Functions in a Transformer

Transformers are usually build with [`VectorSoftmax`](@ref) as activation function, meaning that the activation function looks at each columns of a matrix (or tensor) independently:

```math
\mathrm{softmax}([v^1, v^2, \ldots, v^n]) \equiv \left[ \frac{e^{v^1_i}}{\sum_{i'=1}^de^{v^1_{i'}}}, \frac{e^{v^2_i}}{\sum_{i'=1}^de^{v^2_{i'}}}, \ldots, \frac{e^{v^n_i}}{\sum_{i'=1}^de^{v^n_{i'}}} \right].
```

One can however also use a [`MatrixSoftmax`](@ref):

```math
\mathrm{Msoftmax}(V) \equiv \frac{e^{V_{ij}}}{\sum_{i,j}e^{V_ij}}.
```

```@example softmax_comparison
using GeometricMachineLearning

act1 = GeometricMachineLearning.VectorSoftmax()
act2 = GeometricMachineLearning.MatrixSoftmax()

A = [1 2 3; 1 2 3; 1 2 3]
```

```@example softmax_comparison
act1(A)
```

```@example softmax_comparison
act2(A)
```

We can now train transformers with these different activation functions in the [`MultiHeadAttention`](@ref) layers:

```@example softmax_comparison
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate # hide
using LaTeXStrings # hide
using CairoMakie  # hide
CairoMakie.activate!() # hide
import Random # hide
Random.seed!(123) # hide

const timestep = .3
const n_init_con = 5

# ensemble problem
ep = hodeensemble([rand(2) for _ in 1:n_init_con], [rand(2) for _ in 1:n_init_con]; timestep = timestep)
dl = DataLoader(integrate(ep, ImplicitMidpoint()); suppress_info = true)

nothing # hide
```

We now define the architectures and train them: 

```@example softmax_comparison
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

loss_array1 = o1(nn1, dl, batch, n_epochs; show_progress = false)
loss_array2 = o2(nn2, dl, batch, n_epochs; show_progress = false)

nothing # hide
```

```@setup softmax_comparison
morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256) # hide
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256) # hide

function plot_training_losses(loss_array_vector_softmax, loss_array_matrix_softmax; theme = :dark)
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
        xlabel="Epoch", 
        ylabel="Training Loss",
        xlabelcolor = textcolor,
        ylabelcolor = textcolor,
        yscale = log10
    )
    lines!(ax, loss_array_vector_softmax, color = mpurple, label = "VecSoftM")
    lines!(ax, loss_array_matrix_softmax,  color = mred, label = "MatSoftM")
    axislegend(; position = (.82, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

fig_dark, ax_dark = plot_training_losses(loss_array1, loss_array2; theme = :dark)
fig_light, ax_light = plot_training_losses(loss_array1, loss_array2; theme = :light)

save("softmax_comparison_dark.png", fig_dark; px_per_unit = 1.2)
save("softmax_comparison_light.png", fig_light; px_per_unit = 1.2)

nothing
```

![Training loss for the different networks.](softmax_comparison_light.png)
![Training loss for the different networks.](softmax_comparison_dark.png)


We further evaluate a trajectory with the trained networks for 300 time steps: 

```@setup softmax_comparison
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
save("softmax_comparison_validation_light.png", fig_light; px_per_unit = 1.2)
save("softmax_comparison_validation_dark.png", fig_dark; px_per_unit = 1.2)

nothing
```

![Validation of the different networks.](softmax_comparison_validation_light.png)
![Validation of the different networks.](softmax_comparison_validation_dark.png)