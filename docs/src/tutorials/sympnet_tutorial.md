# SympNets with `GeometricMachineLearning`

This page serves as a short introduction into using [SympNets](@ref "SympNet Architecture") with `GeometricMachineLearning`. 

```@eval
Main.remark(raw"As with any neural network we have to make the following choices:
" * Main.indentation * raw"1. specify the *architecture*,
" * Main.indentation * raw"2. specify the *type* and *backend*,
" * Main.indentation * raw"3. pick an *optimizer* for training the network,
" * Main.indentation * raw"4. specify how you want to perform *batching*,
" * Main.indentation * raw"5. choose a number of epochs,
" * Main.indentation * raw"where points 1 and 3 depend on a variable number of hyperparameters.")
```
For the SympNet point 1 is done by calling [`LASympNet`](@ref) or [`GSympNet`](@ref), point 2 is done by calling `NeuralNetwork`, point 3 is done by calling [`Optimizer`](@ref) and point 4 is done by calling [`Batch`](@ref).

## Loss function

The [`FeedForwardLoss`](@ref) is the default choice used in `GeometricMachineLearning` for training SympNets, this [can however by altered](@ref "Adjusting the Loss Function").

## Training a Harmonic Oscillator

Let us begin with a simple example, the pendulum system, the Hamiltonian of which is 
```math
H:(q,p)\in\mathbb{R}^2 \mapsto \frac{1}{2}p^2-cos(q) \in \mathbb{R}.
```

Here we take the ODE from [`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl) and integrate it with `GeometricIntegrators` [Kraus:2020:GeometricIntegrators](@cite):

```@example sympnet
import GeometricProblems.HarmonicOscillator as ho
using GeometricIntegrators: ImplicitMidpoint, integrate
import Random # hide

Random.seed!(123) # hide

# the problem is the ODE of the harmonic oscillator
problem = ho.hodeproblem(; tspan = 500)

# integrate the system
solution = integrate(problem, ImplicitMidpoint())

nothing # hide
```

We can then conveniently handle the data with [the data loader](@ref "The Data Loader"):

```@example sympnet
using GeometricMachineLearning
# we can conveniently handle the data with 
dl_raw = DataLoader(solution)

nothing # hide
```

Note that we have not yet specified the type and backend that we want to use[^1]. We do this now:

[^1]: Note that we also have to reallocate the data for [`DataLoader`](@ref) in this case to conform with the neural network parameters. 

```@example sympnet
# specify the data type and the backend
type = Float16
backend = CPU()

dl = DataLoader(dl_raw, backend, type)

nothing # hide
```

Next we specify the architectures. `GeometricMachineLearning` provides useful defaults for all parameters although they can be specified manually (which is done in the following):

```@example sympnet
# layer dimension for gradient module 
const upscaling_dimension = 3

# hidden layers
const nhidden = 1

# activation function
const activation = tanh

# number of layers for the G-SympNet
const n_layers = 2

# number of linear layers in each "linear block" in the LA-SympNet
const depth = 2

# calling G-SympNet architecture 
gsympnet = GSympNet(dl, upscaling_dimension=upscaling_dimension, n_layers=n_layers, activation=activation)

# calling LA-SympNet architecture 
lasympnet = LASympNet(dl, nhidden=nhidden, activation=activation, depth = depth)

# initialize the networks
la_nn = NeuralNetwork(lasympnet, backend, type) 
g_nn = NeuralNetwork(gsympnet, backend, type)

nothing # hide
```

If we want to obtain information on the number of parameters in a neural network, we can do that with the function `parameterlength`. For the [`LASympNet`](@ref):
```@example sympnet
parameterlength(la_nn.model)
```

And for the [`GSympNet`](@ref):
```@example sympnet
parameterlength(g_nn.model)
```

```@eval
Main.remark(raw"We can also specify whether we would like to start with a layer that changes the ``q``-component or one that changes the ``p``-component. This can be done via the keywords `init_upper` for `GSympNet`, and `init_upper_linear` and `init_upper_act` for `LASympNet`.")
```

We have to define an optimizer which will be use in the training of the SympNet. For more details on optimizer, please see the [corresponding documentation](@ref "Neural Network Optimizers"). In this example we use [Adam](@ref "The Adam Optimizer"):

```@example sympnet
# set up optimizer; for this we first need to specify the optimization method
opt_method = AdamOptimizer(type)
# we then call the optimizer struct which allocates the cache
la_opt = Optimizer(opt_method, la_nn)
g_opt = Optimizer(opt_method, g_nn)

nothing # hide
```

We can now perform the training of the neural networks:

```@example sympnet
# determine the batch size (the number of samples in one batch)
const batch_size = 16

batch = Batch(batch_size)

# number of training epochs
const nepochs = 100

# perform training (returns array that contains the total loss for each training step)
g_loss_array = g_opt(g_nn, dl, batch, nepochs; show_progress = false)
la_loss_array = la_opt(la_nn, dl, batch, nepochs; show_progress = false)
nothing # hide
```

We can also plot the training errors against the epoch (here the ``y``-axis is in log-scale):
```@example sympnet
using CairoMakie
using LaTeXStrings

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256) # hide

function make_error_plot(; theme = :dark) # hide
textcolor = theme == :dark ? :white : :black # hide
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
    ylabel="Training loss",
    xlabelcolor = textcolor,
    ylabelcolor = textcolor,
    yscale = log10
    )

lines!(ax, g_loss_array, label=L"$G$-SympNet", color=morange)
lines!(ax, la_loss_array, label=L"$LA$-SympNet", color=mpurple)
axislegend(; position = (.82, .75), backgroundcolor = :transparent, labelcolor = textcolor) # hide
fig_name = theme == :dark ? "sympnet_training_loss_dark.png" : "sympnet_training_loss.png" # hide
save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_error_plot(; theme = :dark) # hide
make_error_plot(; theme = :light) # hide

Main.include_graphics("sympnet_training_loss") # hide
```

Now we can make a prediction. Let's compare the initial data with a prediction starting from the same phase space point using the function [`GeometricMachineLearning.iterate`](@ref):

```@example sympnet
ics = (q=dl.input.q[:, 1, 1], p=dl.input.p[:, 1, 1])

steps_to_plot = 200

#predictions
la_trajectory = iterate(la_nn, ics; n_points = steps_to_plot)
g_trajectory =  iterate(g_nn, ics; n_points = steps_to_plot)

function make_prediction_plot(; theme = :dark) # hide
textcolor = theme == :dark ? :white : :black # hide
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
    xlabel=L"q", 
    ylabel=L"p",
    xlabelcolor = textcolor,
    ylabelcolor = textcolor
    )

lines!(ax, dl.input.q[1, 1:steps_to_plot, 1], dl.input.p[1, 1:steps_to_plot, 1], label="training data", color = mblue)
lines!(ax, la_trajectory.q[1, :], la_trajectory.p[1, :], label=L"$LA$-Sympnet", color = mpurple)
lines!(ax, g_trajectory.q[1, :], g_trajectory.p[1, :], label=L"$G$-Sympnet", color = morange)
axislegend(; position = (.82, .75), backgroundcolor = :transparent, labelcolor = textcolor) # hide
fig_name = theme == :dark ? "sympnet_prediction_dark.png" : "sympnet_prediction.png" # hide
save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_prediction_plot(; theme = :dark) # hide
make_prediction_plot(; theme = :light) # hide

Main.include_graphics("sympnet_prediction") # hide
```

We see that [`GSympNet`](@ref) outperforms [`LASympNet`](@ref) on this problem.

```@eval
Main.remark(raw"We have actually never observed a scenario in which the ``LA``-SympNet can outperform the ``G``-SympNet. The ``G``-SympNet seems to train faster, be more accurate and less sensitive to the chosen hyperparameters and initialization of the weights. They are also more straightforward to interpret. We therefore use the ``G``-SympNet as a basis for the *linear symplectic transformer.*")
```