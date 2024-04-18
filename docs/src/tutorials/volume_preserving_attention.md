# Comparison of different `VolumePreservingAttention`

In the [section of volume-preserving attention](../layers/attention_layer.md) we mentioned two ways of computing volume-preserving attention: one where we compute the correlations with a skew-symmetric matrix and one where we compute the correlations with an arbitrary matrix. Here we compare the two approaches. When calling the `VolumePreservingAttention` layer we can specify whether we want to use the skew-symmetric or the arbitrary weighting by setting the keyword `skew_sym = true` and `skew_sym = false` respectively. 

In here we demonstrate the differences between the two approaches for computing correlations. For this we first generate a training set consisting of two collections of curves: (i) sine curves and (ii) cosine curve. 

```@example volume_preserving_attention
using GeometricMachineLearning # hide
using GeometricMachineLearning: FeedForwardLoss, TransformerLoss # hide
using Plots # hide
import Random # hide 
Random.seed!(123) # hide

sine_cosine = zeros(1, 1000, 2)
sine_cosine[1, :, 1] .= sin.(0.:.1:99.9)
sine_cosine[1, :, 2] .= cos.(0.:.1:99.9)


const dl = DataLoader(Float16.(sine_cosine))
```

The third axis (i.e. the parameter axis) has length two, meaning we have two different kinds of curves: 

```@example volume_preserving_attention
plot(dl.input[1, :, 1], label = "sine")
plot!(dl.input[1, :, 2], label = "cosine")
```

We want to train a single neural network on both these curves. We compare three networks which are of the following form: 

```math
\mathtt{network} = \mathcal{NN}_d\circ\Psi\circ\mathcal{NN}_u,
```

where ``\mathcal{NN}_u`` refers to a neural network that scales up and ``\mathcal{NN}_d`` refers to a neural network that scales down. The up and down scaling is done with simple dense layers: 

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

const T = Float16

function set_up_networks(upscale_dimension::Int = upscale_dimension_1)
    model_skew = Chain(Dense(1, upscale_dimension, tanh), VolumePreservingAttention(upscale_dimension, seq_length; skew_sym = true),  Dense(upscale_dimension, 1, identity; use_bias = true))
    model_arb  = Chain(Dense(1, upscale_dimension, tanh), VolumePreservingAttention(upscale_dimension, seq_length; skew_sym = false), Dense(upscale_dimension, 1, identity; use_bias = true))
    model_comp = Chain(Dense(1, upscale_dimension, tanh), Dense(upscale_dimension, 1, identity; use_bias = true))

    nn_skew = NeuralNetwork(model_skew, CPU(), T)
    nn_arb  = NeuralNetwork(model_arb,  CPU(), T)
    nn_comp = NeuralNetwork(model_comp, CPU(), T)

    nn_skew, nn_arb, nn_comp
end

nn_skew, nn_arb, nn_comp = set_up_networks()
```

We expect the third network to not be able to learn anything useful since it cannot resolve time series data: a regular feedforward network only ever sees one datum at a time. 

Next we train the networks (here we pick a batch size of 30):

```@example volume_preserving_attention
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
    loss_array_skew = o_skew(nn_skew, dl, batch, n_epochs, TransformerLoss(batch))
    loss_array_arb  = o_arb( nn_arb,  dl, batch, n_epochs, TransformerLoss(batch))
    loss_array_comp = o_comp(nn_comp, dl, batch2, n_epochs, FeedForwardLoss())

    loss_array_skew, loss_array_arb, loss_array_comp
end

loss_array_skew, loss_array_arb, loss_array_comp = train_networks!(nn_skew, nn_arb, nn_comp)

function plot_training_losses(loss_array_skew, loss_array_arb, loss_array_comp)
    p = plot(loss_array_skew, color = 2, label = "skew", yaxis = :log)
    plot!(p, loss_array_arb,  color = 3, label = "arb")
    plot!(p, loss_array_comp, color = 4, label = "comp")

    p
end

plot_training_losses(loss_array_skew, loss_array_arb, loss_array_comp)
```

Looking at the training errors, we can see that the network with the skew-symmetric weighting is stuck at a relatively high error rate, whereas the loss for  the network with the arbitrary weighting is decreasing to a significantly lower level. The feedforward network without the attention mechanism is not able to learn anything useful (as was expected). 

The following demonstrates the predictions of our approaches[^1]: 

[^1]: Here we have to use the architectures `DummyTransformer` and `DummyNNIntegrator` to reformulate the three neural networks defined here as `NeuralNetworkIntegrator`s. Normally the user should try to use predefined architectures in `GeometricMachineLearning`, that way they never use `DummyTransformer` and `DummyNNIntegrator`. 

```@example volume_preserving_attention
initial_condition = dl.input[:, 1:seq_length, 2]

function make_networks_neural_network_integrators(nn_skew, nn_arb, nn_comp)
    nn_skew = NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), nn_skew.model, nn_skew.params)
    nn_arb  = NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), nn_arb.model,  nn_arb.params)
    nn_comp = NeuralNetwork(GeometricMachineLearning.DummyNNIntegrator(), nn_comp.model, nn_comp.params)

    nn_skew, nn_arb, nn_comp
end

nn_skew, nn_arb, nn_comp = make_networks_neural_network_integrators(nn_skew, nn_arb, nn_comp)

function produce_validation_plot(n_points::Int, nn_skew = nn_skew, nn_arb = nn_arb, nn_comp = nn_comp; initial_condition::Matrix=initial_condition, type = :cos)
    validation_skew = iterate(nn_skew, initial_condition; n_points = n_points, prediction_window = 1)
    validation_arb  = iterate(nn_arb,  initial_condition; n_points = n_points, prediction_window = 1)
    validation_comp = iterate(nn_comp, initial_condition[:, 1]; n_points = n_points)

    p2 = type == :cos ? plot(dl.input[1, 1:n_points, 2], color = 1, label = "reference") : plot(dl.input[1, 1:n_points, 1], color = 1, label = "reference")

    plot!(validation_skew[1, :], color = 2, label = "skew")
    plot!(p2, validation_arb[1, :], color = 3, label = "arb")
    plot!(p2, validation_comp[1, :], color = 4, label = "comp")
    vline!([seq_length], color = :red, label = "start of prediction")

    p2 
end

p2 = produce_validation_plot(40)
```
In the above plot we can see that the network with the arbitrary weighting performs much better; even though the green line does not fit the blue line very well either, it manages to least qualitatively reflect the training data.  We can also plot the predictions for longer time intervals: 

```@example volume_preserving_attention 
p3 = produce_validation_plot(400)
``` 

We can also plot the comparison with the sine function: 

```@example volume_preserving_attention 
initial_condition = dl.input[:, 1:seq_length, 1]

p2 = produce_validation_plot(40, initial_condition = initial_condition, type = :sin)
```

This advantage of the volume-preserving attention with arbitrary weighting may however be due to the fact that the skew-symmetric attention only has 3 learnable parameters, as opposed to 9 for the arbitrary weighting. If we increase the *upscaling dimension* the result changes: 

```@example volume_preserving_attention
const upscale_dimension_2 = 10

nn_skew, nn_arb, nn_comp = set_up_networks(upscale_dimension_2)

o_skew, o_arb, o_comp = set_up_optimizers(nn_skew, nn_arb, nn_comp)

loss_array_skew, loss_array_arb, loss_array_comp = train_networks!(nn_skew, nn_arb, nn_comp)

plot_training_losses(loss_array_skew, loss_array_arb, loss_array_comp)
```

```@example volume_preserving_attention 
initial_condition = dl.input[:, 1:seq_length, 2]

nn_skew, nn_arb, nn_comp = make_networks_neural_network_integrators(nn_skew, nn_arb, nn_comp)

p2 = produce_validation_plot(40, nn_skew, nn_arb, nn_comp)
```

And for a longer time interval: 

```@example volume_preserving_attention
p3 = produce_validation_plot(200, nn_skew, nn_arb, nn_comp)
```