# Comparing Different `VolumePreservingAttention` Mechanisms

In the [section on volume-preserving attention](@ref "Volume-Preserving Attention") we mentioned two ways of computing volume-preserving attention: one where we compute the correlations with a skew-symmetric matrix and one where we compute the correlations with an arbitrary matrix. Here we compare the two approaches. When calling the [`VolumePreservingAttention`](@ref) layer we can specify whether we want to use the skew-symmetric or the arbitrary weighting by setting the keyword `skew_sym = true` and `skew_sym = false` respectively. 

In here we demonstrate the differences between the two approaches for computing correlations. For this we first generate a training set consisting of two collections of curves: (i) sine curves and (ii) cosine curve. 

```@example volume_preserving_attention
using GeometricMachineLearning # hide
using GeometricMachineLearning: FeedForwardLoss, TransformerLoss, params # hide
import Random # hide
Random.seed!(123) # hide
sine_cosine = zeros(1, 1000, 2)
sine_cosine[1, :, 1] .= sin.(0.:.1:99.9)
sine_cosine[1, :, 2] .= cos.(0.:.1:99.9)

const T = Float16
const dl = DataLoader(T.(sine_cosine); suppress_info = true)

nothing # hide
```

The third axis (i.e. the parameter axis) has length two, meaning we have two different kinds of curves, i.e. the data look like this:

```@setup volume_preserving_attention
using CairoMakie, LaTeXStrings # hide
morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256) # hide
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256) # hide
function make_comparison_plot(; theme = :dark) # hide
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
    xlabel=L"t", 
    ylabel=L"z",
    xlabelcolor = textcolor,
    ylabelcolor = textcolor,
    )

lines!(ax, dl.input[1, 1:200, 1], label=L"\sin(t)", color = morange)
lines!(ax, dl.input[1, 1:200, 2], label=L"\cos(t)", color = mpurple)
axislegend(; position = (.82, .75), backgroundcolor = theme == :dark ? :transparent : :white, labelcolor = textcolor) # hide
fig_name = theme == :dark ? "curve_comparison_dark.png" : "curve_comparison_light.png" # hide
save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_comparison_plot(; theme = :dark) # hide
make_comparison_plot(; theme = :light) # hide

nothing
```

![The data we treat here contains two different curves.](curve_comparison_light.png)
![The data we treat here contains two different curves.](curve_comparison_dark.png)

We want to train a single neural network on both these curves. We already noted [before](@ref "Why use Transformers for Model Order Reduction") that a simple feedforward neural network cannot do this. Here we compare three networks which are of the following form: 

```math
\mathtt{network} = \mathcal{NN}_d\circ\Psi\circ\mathcal{NN}_u,
```

where ``\mathcal{NN}_u`` refers to a neural network that [scales up](@ref "The Upscaling") and ``\mathcal{NN}_d`` refers to a neural network that scales down. The up and down scaling is done with simple dense layers: 

```math
\mathcal{NN}_u(x) = \mathrm{tanh}(a_ux + b_u) \text{ and } \mathcal{NN}_d(x) = a_d^Tx + b_d,
```
where ``a_u, b_u, a_d\in\mathbb{R}^\mathrm{ud}`` and ``b_d`` is a scalar. `ud` refers to *upscaling dimension*. For ``\Psi`` we consider three different choices:
1. a volume-preserving attention with skew-symmetric weighting,
2. a volume-preserving attention with arbitrary weighting,
3. an identity layer.

We further choose a sequence length 5 (i.e. the network always sees the last 5 time steps) and always predict one step into the future (i.e. the prediction window is set to 1):

```@example volume_preserving_attention
const seq_length = 3
const prediction_window = 1
const upscale_dimension_1 = 2

function set_up_networks(upscale_dimension::Int = upscale_dimension_1)
    model_skew = Chain( Dense(1, upscale_dimension, tanh), 
                        VolumePreservingAttention(upscale_dimension, seq_length; skew_sym = true),
                        Dense(upscale_dimension, 1, identity; use_bias = true)
                        )

    model_arb  = Chain( Dense(1, upscale_dimension, tanh), 
                        VolumePreservingAttention(upscale_dimension, seq_length; skew_sym = false), 
                        Dense(upscale_dimension, 1, identity; use_bias = true)
                        )

    model_comp = Chain( Dense(1, upscale_dimension, tanh), 
                        Dense(upscale_dimension, 1, identity; use_bias = true)
                        )

    nn_skew = NeuralNetwork(model_skew, CPU(), T)
    nn_arb  = NeuralNetwork(model_arb,  CPU(), T)
    nn_comp = NeuralNetwork(model_comp, CPU(), T)

    nn_skew, nn_arb, nn_comp
end

nn_skew, nn_arb, nn_comp = set_up_networks()
nothing # hide
```

We expect the third network to not be able to learn anything useful since it cannot resolve time series data: a regular feedforward network only ever sees one datum at a time. 

Next we train the networks (here we pick a batch size of 30 and train for 1000 epochs):

```@setup volume_preserving_attention
function set_up_optimizers(nn_skew, nn_arb, nn_comp)
    o_skew = Optimizer(AdamOptimizer(T), nn_skew)
    o_arb  = Optimizer(AdamOptimizer(T), nn_arb)
    o_comp = Optimizer(AdamOptimizer(T), nn_comp)

    o_skew, o_arb, o_comp
end

o_skew, o_arb, o_comp = set_up_optimizers(nn_skew, nn_arb, nn_comp)

const n_epochs = 1000

const batch_size = 30

const batch = Batch(batch_size, seq_length, prediction_window)
const batch2 = Batch(batch_size)

function train_networks!(nn_skew, nn_arb, nn_comp)
    loss_array_skew = o_skew(nn_skew, dl, batch, n_epochs, TransformerLoss(batch); show_progress = false)
    loss_array_arb  = o_arb( nn_arb,  dl, batch, n_epochs, TransformerLoss(batch); show_progress = false)
    loss_array_comp = o_comp(nn_comp, dl, batch2, n_epochs, FeedForwardLoss(); show_progress = false)

    loss_array_skew, loss_array_arb, loss_array_comp
end

loss_array_skew, loss_array_arb, loss_array_comp = train_networks!(nn_skew, nn_arb, nn_comp)

function plot_training_losses(loss_array_skew, loss_array_arb, loss_array_comp; theme = :dark)
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
        ylabel="Training loss",
        xlabelcolor = textcolor,
        ylabelcolor = textcolor,
        yscale = log10
    )
    lines!(ax, loss_array_skew, color = mblue, label = "skew")
    lines!(ax, loss_array_arb,  color = mred, label = "arb")
    lines!(ax, loss_array_comp, color = mgreen, label = "comp")
    axislegend(; position = (.82, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

fig_dark, ax_dark = plot_training_losses(loss_array_skew, loss_array_arb, loss_array_comp; theme = :dark)
fig_light, ax_light = plot_training_losses(loss_array_skew, loss_array_arb, loss_array_comp; theme = :light)
save("training_loss_vpa_light.png", fig_light; px_per_unit = 1.2)
save("training_loss_vpa_dark.png", fig_dark; px_per_unit = 1.2)

nothing
```

![The training losses for the three networks.](training_loss_vpa_light.png)
![The training losses for the three networks.](training_loss_vpa_dark.png)

Looking at the training errors, we can see that the network with the skew-symmetric weighting is stuck at a relatively high error rate, whereas the loss for  the network with the arbitrary weighting is decreasing to a significantly lower level. The feedforward network without the attention mechanism is not able to learn anything useful (as was expected). 

Before we can use the trained neural networks for prediction we have to make them [`TransformerIntegrator`](@ref)s or [`NeuralNetworkIntegrator`](@ref)s[^1]:

[^1]: Here we have to use the architectures [`GeometricMachineLearning.DummyTransformer`](@ref) and [`GeometricMachineLearning.DummyNNIntegrator`](@ref) to reformulate the three neural networks defined here as [`NeuralNetworkIntegrator`](@ref)s or [`TransformerIntegrator`](@ref)s. These *dummy architectures* can be used if the user wants to specify new neural network integrators that are not yet defined in `GeometricMachineLearning`. 

```@example volume_preserving_attention
initial_condition = dl.input[:, 1:seq_length, 2]

function make_networks_neural_network_integrators(nn_skew, nn_arb, nn_comp)
    nn_skew = NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), 
                            nn_skew.model, 
                            params(nn_skew), 
                            CPU())
    nn_arb  = NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), 
                            nn_arb.model,  
                            params(nn_arb), 
                            CPU())
    nn_comp = NeuralNetwork(GeometricMachineLearning.DummyNNIntegrator(), 
                            nn_comp.model, 
                            params(nn_comp), 
                            CPU())

    nn_skew, nn_arb, nn_comp
end

nn_skew, nn_arb, nn_comp = make_networks_neural_network_integrators(nn_skew, nn_arb, nn_comp)

nothing # hide
```

```@setup volume_preserving_attention
function produce_validation_plot_single(n_points::Int, nn_skew = nn_skew, nn_arb = nn_arb, nn_comp = nn_comp; initial_condition::Matrix=initial_condition, type = :cos, theme = :dark)
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
        xlabel=L"t", 
        ylabel=L"z",
        xlabelcolor = textcolor,
        ylabelcolor = textcolor,
        )
    validation_skew = iterate(nn_skew, initial_condition; n_points = n_points, prediction_window = 1)
    validation_arb  = iterate(nn_arb,  initial_condition; n_points = n_points, prediction_window = 1)
    validation_comp = iterate(nn_comp, initial_condition[:, 1]; n_points = n_points)

    p2 = type == :cos ? lines!(dl.input[1, 1:n_points, 2], color = mpurple, label = "reference") : plot(dl.input[1, 1:n_points, 1], color = mpurple, label = "reference")

    lines!(ax, validation_skew[1, :], color = mblue, label = "skew")
    lines!(ax, validation_arb[1, :], color = mred, label = "arb")
    lines!(ax, validation_comp[1, :], color = mgreen, label = "comp")
    vlines!(ax, [seq_length], color = mred, label = "start of prediction")

    axislegend(; position = (.82, .75), backgroundcolor = theme == :dark ? :transparent : :white, labelcolor = textcolor)
    fig, ax
end

function produce_validation_plot(n_points::Int, nn_skew = nn_skew, nn_arb = nn_arb, nn_comp = nn_comp; initial_condition::Matrix=initial_condition, type = :cos)
    fig_dark, ax_dark = produce_validation_plot_single(n_points, nn_skew, nn_arb, nn_comp; initial_condition = initial_condition, type = type, theme = :dark)
    fig_light, ax_light = produce_validation_plot_single(n_points, nn_skew, nn_arb, nn_comp; initial_condition = initial_condition, type = type, theme = :light)

    fig_dark, fig_light, ax_dark, ax_light
end

nothing
```

```@example volume_preserving_attention
fig_dark, fig_light, ax_dark, ax_light  = produce_validation_plot(40) # hide
save("plot40_dark.png", fig_dark; px_per_unit = 1.2) # hide
save("plot40_light.png", fig_light; px_per_unit = 1.2) # hide
nothing
```

![Comparing the two volume-preserving attention mechanisms for 40 points.](plot40_light.png)
![Comparing the two volume-preserving attention mechanisms for 40 points.](plot40_dark.png)

In the plot above we can see that the network with the arbitrary weighting performs much better; even though the red line does not fit the purple line perfectly, it manages to least qualitatively reflect the training data.  We can also plot the predictions for longer time intervals: 

```@example volume_preserving_attention 
fig_dark, fig_light, ax_dark, ax_light  = produce_validation_plot(400) # hide
save("plot400_dark.png", fig_dark; px_per_unit = 1.2) # hide
save("plot400_light.png", fig_light; px_per_unit = 1.2) # hide
nothing # hide
```

![Comparing the two volume-preserving attention mechanisms for 400 points.](plot400_light.png)
![Comparing the two volume-preserving attention mechanisms for 400 points.](plot400_dark.png)

This advantage of the volume-preserving attention with arbitrary weighting may however be due to the fact that the skew-symmetric attention only has 3 learnable parameters, as opposed to 9 for the arbitrary weighting. We can increase the *upscaling dimension* and see how it affects the result: 

```@example volume_preserving_attention
const upscale_dimension_2 = 10

nn_skew, nn_arb, nn_comp = set_up_networks(upscale_dimension_2)

o_skew, o_arb, o_comp = set_up_optimizers(nn_skew, nn_arb, nn_comp)

loss_array_skew, loss_array_arb, loss_array_comp = train_networks!(nn_skew, nn_arb, nn_comp) # hide
fig_dark, ax_dark = plot_training_losses(loss_array_skew, loss_array_arb, loss_array_comp; theme = :dark) # hide
fig_light, ax_light = plot_training_losses(loss_array_skew, loss_array_arb, loss_array_comp; theme = :light) # hide
save("training_loss2_vpa_light.png", fig_light; px_per_unit = 1.2) # hide
save("training_loss2_vpa_dark.png", fig_dark; px_per_unit = 1.2) # hide
nothing # hide
```

![Comparison for 40 points, but with an upscaling of ten.](training_loss2_vpa_light.png)
![Comparison for 40 points, but with an upscaling of ten.](training_loss2_vpa_dark.png)

```@example volume_preserving_attention 
initial_condition = dl.input[:, 1:seq_length, 2]

nn_skew, nn_arb, nn_comp = make_networks_neural_network_integrators(nn_skew, nn_arb, nn_comp)

fig_dark, fig_light, ax_dark, ax_light = produce_validation_plot(40, nn_skew, nn_arb, nn_comp)

save("plot40_sine2_dark.png", fig_dark; px_per_unit = 1.2) # hide
save("plot40_sine2_light.png", fig_light; px_per_unit = 1.2) # hide
nothing # hide
```

![](plot40_sine2_light.png)
![](plot40_sine2_dark.png)

And for a longer time interval: 

```@example volume_preserving_attention
fig_dark, fig_light, ax_dark, ax_light = produce_validation_plot(200, nn_skew, nn_arb, nn_comp)


save("plot200_sine2_dark.png", fig_dark; px_per_unit = 1.2) # hide
save("plot200_sine2_light.png", fig_light; px_per_unit = 1.2) # hide
nothing
```

![](plot200_sine2_light.png)
![](plot200_sine2_dark.png)

Here we see that the arbitrary weighting quickly fails and the skew-symmetric weighting performs better on longer time scales.

## Library Functions

```@docs
GeometricMachineLearning.DummyNNIntegrator
GeometricMachineLearning.DummyTransformer
```

```@raw latex
\section*{Chapter Summary}

In this chapter we showed concrete examples of how to improve transformer neural networks by imbuing them with structure. The two examples we gave were (i) enforcing orthogonality constraints for some of the weights in a vision transformer (i.e. putting some of the weights on the \textit{Stiefel manifold}) and (ii) using a volume-preserving transformer to learn the dynamics of a rigid body. In both cases we observed big improvements over the standard transformer that does not consider structure. In the first case the network was not able to learn anything if orthogonality constraints were not imposed and in the second case we obtained greatly improved long-time performance. At the end we also compared two different approaches to designing the volume-preserving transformer: computing correlations based on a \textit{skew-symmetric weighting} and \textit{computing correlations based on an arbitrary weighting}. We saw that often the arbitrary weighting should be preferred over the skew-symmetric weighting, but the arbitrary weighting may also fail in other cases.
```

```@raw html
<!--
```

# References

```@bibliography
Pages = []
Canonical = false

brantner2023generalizing
brantner2024volume
```

```@raw html
-->
```