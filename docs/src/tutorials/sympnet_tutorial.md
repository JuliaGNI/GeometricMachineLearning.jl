```@raw latex
In this chapter we use neural networks as symplectic integrators. We first show SympNets as an example of a neural network-based one-step method and then show linear symplectic transformers as an example of a symplectic multi-step method.
```

# SympNets with `GeometricMachineLearning`

This page serves as a short introduction into using [SympNets](@ref "SympNet Architecture") with `GeometricMachineLearning`. 


## Loss function

The `FeedForwardLoss` is the default choice used in `GeometricMachineLearning` for training SympNets, this can however be changed or [tweaked](@ref "Adjusting the Loss Function").

## Training a Harmonic Oscillator

Here we begin with a simple example, the harmonic oscillator, the Hamiltonian of which is 
```math
H:(q,p)\in\mathbb{R}^2 \mapsto \frac{1}{2}p^2 + \frac{1}{2}q^2 \in \mathbb{R}.
```

Here we take the ODE from [`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl) and integrate it with `GeometricIntegrators` [Kraus:2020:GeometricIntegrators](@cite):

```@example sympnet
import GeometricProblems.HarmonicOscillator as ho
using GeometricIntegrators: ImplicitMidpoint, integrate
import Random # hide
Random.seed!(1234) # hide

# the problem is the ODE of the harmonic oscillator
ho_problem = ho.hodeproblem(; timespan = 500)

# integrate the system
solution = integrate(ho_problem, ImplicitMidpoint())

nothing # hide
```

We call [`DataLoader`](@ref) in order to conveniently handle the data:

```@example sympnet
using GeometricMachineLearning # hide
dl_raw = DataLoader(solution; suppress_info = true)
nothing # hide
```

We have not yet specified the type and backend that we want to use. We do this now:

```@example sympnet
# specify the data type and the backend
type = Float16
backend = CPU()

# we can then make a new instance of `DataLoader` with this backend and type.
dl = DataLoader(dl_raw, backend, type)
nothing # hide
```

Next we specify the architectures[^1]: 

[^1]: `GeometricMachineLearning` provides useful defaults for all parameters, but they can still be specified manually; which is what we are doing here. Details on these parameters can be found in the docstrings for [`GSympNet`](@ref) and [`LASympNet`](@ref).

```@example sympnet
const upscaling_dimension = 2
const nhidden = 1
const activation = tanh
const n_layers = 4 # number of layers for the G-SympNet
const depth = 4 # number of layers in each linear block in the LA-SympNet

# calling G-SympNet architecture 
gsympnet = GSympNet(dl; upscaling_dimension = upscaling_dimension, 
                        n_layers = n_layers, 
                        activation = activation)

# calling LA-SympNet architecture 
lasympnet = LASympNet(dl;   nhidden = nhidden, 
                            activation = activation, 
                            depth = depth)

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
Main.remark(raw"We can also specify whether we would like to start with a layer that changes the ``q``-component or one that changes the ``p``-component. This can be done via the keywords `init_upper` for the `GSympNet`, and `init_upper_linear` and `init_upper_act` for the `LASympNet`.")
```

We have to define an [optimizer](@ref "Standard Neural Network Optimizers") which will be used in training of the SympNet. In this example we use [Adam](@ref "The Adam Optimizer"):

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

We plot the training errors against the epoch (here the ``y``-axis is in log-scale):
```@setup sympnet
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
fig_name = theme == :dark ? "sympnet_training_loss_dark.png" : "sympnet_training_loss_light.png" # hide
save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_error_plot(; theme = :dark) # hide
make_error_plot(; theme = :light) # hide
nothing # hide
```

![](sympnet_training_loss_light.png)
![](sympnet_training_loss_dark.png)

Now we can make a prediction. We compare the initial data with a prediction starting from the same phase space point using the function [`GeometricMachineLearning.iterate`](@ref):

```@example sympnet
ics = (q=dl.input.q[:, 1, 1], p=dl.input.p[:, 1, 1])

steps_to_plot = 200

#predictions
la_trajectory = iterate(la_nn, ics; n_points = steps_to_plot)
g_trajectory =  iterate(g_nn, ics; n_points = steps_to_plot)
nothing # hide
```

We now plot the result:

```@setup sympnet
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
axislegend(; position = (.82, .45), backgroundcolor = :transparent, labelcolor = textcolor) # hide
fig_name = theme == :dark ? "sympnet_prediction_dark.png" : "sympnet_prediction_light.png" # hide
save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_prediction_plot(; theme = :dark) # hide
make_prediction_plot(; theme = :light) # hide
nothing # hide
```

![](sympnet_prediction_light.png)
![](sympnet_prediction_dark.png)

We see that [`GSympNet`](@ref) outperforms [`LASympNet`](@ref) on this problem; the blue line (reference) and the orange line (``G``-SympNet) are in fact almost indistinguishable.

```@eval
Main.remark(raw"We have actually never observed a scenario in which the ``LA``-SympNet can outperform the ``G``-SympNet. The ``G``-SympNet seems usually trains faster, is more accurate and less sensitive to the chosen hyperparameters and initialization of the weights. They are also more straightforward to interpret. We therefore use the ``G``-SympNet as a basis for the *linear symplectic transformer.*")
```

## Comparison with a ResNet

We want to show the advantages of using a SympNet over a standard ResNet that is not symplectic. For this we make a ResNet with a similar size of parameters as the two SympNets have:

```@example sympnet
Random.seed!(1234) # hide
resnet = ResNet(dl, n_layers ÷ 2; activation = activation)

rn_nn = NeuralNetwork(resnet, backend, type)

parameterlength(rn_nn)
```

We now train the network ``\ldots``

```@example sympnet
rn_opt = Optimizer(opt_method, rn_nn)

rn_loss_array = rn_opt(rn_nn, dl, batch, nepochs; show_progress = false)
nothing # hide
```

and plot the loss:

```@setup sympnet
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
lines!(ax, rn_loss_array, label="ResNet", color=mred)
axislegend(; position = (.82, .75), backgroundcolor = :transparent, labelcolor = textcolor) # hide
fig_name = theme == :dark ? "sympnet_resnet_training_loss_dark.png" : "sympnet_resnet_training_loss_light.png" # hide
save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_error_plot(; theme = :dark) # hide
make_error_plot(; theme = :light) # hide
nothing # hide
```

![](sympnet_resnet_training_loss_light.png)
![](sympnet_resnet_training_loss_dark.png)

And we see that the loss is significantly lower than for the ``LA``-SympNet, but slightly higher than for the ``G``-SympNet. We can also plot the prediction:

```@example sympnet
rn_trajectory = iterate(rn_nn, ics; n_points = steps_to_plot)
nothing # hide
```

```@setup sympnet
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
lines!(ax, rn_trajectory.q[1, :], rn_trajectory.p[1, :], label="ResNet", color = mred)
axislegend(; position = (.82, .45), backgroundcolor = :transparent, labelcolor = textcolor) # hide
fig_name = theme == :dark ? "resnet_sympnet_prediction_dark.png" : "resnet_sympnet_prediction_light.png" # hide
save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_prediction_plot(; theme = :dark) # hide
make_prediction_plot(; theme = :light) # hide
nothing # hide
```

![](resnet_sympnet_prediction_light.png)
![](resnet_sympnet_prediction_dark.png)

We see that the ResNet is slowly gaining energy which consitutes unphysical behaviour. If we let this simulation run for even longer, this effect gets more pronounced:

```@example sympnet
steps_to_plot = 800

#predictions
la_trajectory = iterate(la_nn, ics; n_points = steps_to_plot)
g_trajectory =  iterate(g_nn, ics; n_points = steps_to_plot)
rn_trajectory = iterate(rn_nn, ics; n_points = steps_to_plot)
nothing # hide
```

```@setup sympnet
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
lines!(ax, rn_trajectory.q[1, :], rn_trajectory.p[1, :], label="ResNet", color = mred)
axislegend(; position = (.99, .9), backgroundcolor = :transparent, labelcolor = textcolor) # hide
fig_name = theme == :dark ? "resnet_sympnet_prediction_long_dark.png" : "resnet_sympnet_prediction_long_light.png" # hide
save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_prediction_plot(; theme = :dark) # hide
make_prediction_plot(; theme = :light) # hide
nothing # hide
```

![](resnet_sympnet_prediction_long_light.png)
![](resnet_sympnet_prediction_long_dark.png)

The behavior the ResNet exhibits is characteristic of integration schemes that do not preserve structure: the error in a single time step can be made very small, but for long-time simulations one typically has to consider symplecticity or other properties. Also note that the curves produced by the ``LA``-SympNet and the ``G``-SympNet are closed (or nearly closed). This is a property of symplectic maps in two dimensions that is preserved by construction [hairer2006geometric](@cite).