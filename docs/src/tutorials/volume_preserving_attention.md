# Comparison of different `VolumePreservingAttention`

In the [section of volume-preserving attention](../layers/attention_layer.md) we mentioned two forms of computing volume-preserving attention: one where we compute the correlations with a skew-symmetric matrix and one where we compute the correlations with an arbitrary matrix. Here we compare the two approaches. They can be called with `skew_sym = true` and `skew_sym = false` respectively. 

We first generate a training set consisting of two collections of curves: (i) sine curves and (ii) cosine curve. 

```@example volume_preserving_attention
using GeometricMachineLearning 
using GeometricMachineLearning: transformer_loss
using Plots # hide
import Random # hide 
Random.seed!(123) # hide

sine_cosine = zeros(1, 1000, 2)
sine_cosine[1, :, 1] .= sin.(0.:.1:99.9)
sine_cosine[1, :, 2] .= cos.(0.:.1:99.9)


dl = DataLoader(Float16.(sine_cosine))
```

Our training data thus consist of two curves: 

```@example volume_preserving_attention
plot(dl.input[1, :, 1], label = "sine")
plot!(dl.input[1, :, 2], label = "cosine")
```

We want to train a single neural network on both these curves. We compare three networks: 
1. A composition of a dense layer, volume-preserving attention with skew-symmetric weighting and another dense layer (with identity activation).
2. A composition of a dense layer, volume-preserving attention with arbitrary weighting and another dense layer (with identity activation).
3. A composition of two dense layers without an attention mechanism.

```@example volume_preserving_attention
model_skew = Chain(Dense(1, 3, tanh), VolumePreservingAttention(3, 5; skew_sym = true),  Dense(3, 1, identity))
model_arb  = Chain(Dense(1, 3, tanh), VolumePreservingAttention(3, 5; skew_sym = false), Dense(3, 1, identity))
model_comp = Chain(Dense(1, 3, tanh), Dense(3, 1, identity))

nn_skew = NeuralNetwork(model_skew, CPU(), Float16)
nn_arb  = NeuralNetwork(model_arb,  CPU(), Float16)
nn_comp = NeuralNetwork(model_comp, CPU(), Float16)
```

We expect the third network to not be able to learn anything useful since it cannot resolve time series data: a regular feedforward network only ever sees one datum at a time. 

Next we train the networks:

```@example volume_preserving_attention
o_skew = Optimizer(AdamOptimizer(), nn_skew)
o_arb  = Optimizer(AdamOptimizer(), nn_arb)
o_comp = Optimizer(AdamOptimizer(), nn_comp)

n_epochs = 150

batch = Batch(100, 5)
batch2 = Batch(100, 1)

loss_array_skew = o_skew(nn_skew, dl, batch, n_epochs, transformer_loss)
loss_array_arb  = o_arb( nn_arb,  dl, batch, n_epochs, transformer_loss)
loss_array_comp = o_comp(nn_comp, dl, batch2, n_epochs)

p = plot(loss_array_skew, color = 2, label = "skew", yaxis = :log)
plot!(p, loss_array_arb,  color = 3, label = "arb")
plot!(p, loss_array_comp, color = 4, label = "comp")
```

Looking at the training errors, we can see that the network with the skew-symmetric weighting is stuck at a relatively high error rate, whereas the loss for  the network with the arbitrary weighting is decreasing to a significantly lower level. The feedforward network without the attention mechanism is not able to learn anything useful. 

The following demonstrates the predictions of our approaches: 

```@example volume_preserving_attention
initial_condition = dl.input[:, 1:5, 2]

nn_skew = NeuralNetwork(GeometricMachineLearning.DummyTransformer(5), nn_skew.model, nn_skew.params)
nn_arb  = NeuralNetwork(GeometricMachineLearning.DummyTransformer(5), nn_arb.model,  nn_arb.params)
nn_comp = NeuralNetwork(GeometricMachineLearning.DummyNNIntegrator(), nn_comp.model, nn_comp.params)

function produce_validation_plot(n_points::Int)
    validation_skew = iterate(nn_skew, initial_condition, n_points = n_points)
    validation_arb  = iterate(nn_arb,  initial_condition, n_points = n_points)
    validation_comp = iterate(nn_comp, initial_condition[:, 1], n_points = n_points)

    p2 = plot(dl.input[1, 1:n_points, 2], color = 1, label = "reference")

    plot!(validation_skew[1, :], color = 2, label = "skew")
    plot!(p2, validation_arb[1, :], color = 3, label = "arb")
    plot!(p2, validation_comp[1, :], color = 4, label = "comp")
    vline!([5], color = :red, label = "start of prediction")

    p2 
end

p2 = produce_validation_plot(50)
```
In the above plot we can see that the network with the arbitrary weighting ostensibly performs better.  We can however also plot the predictions for longer time intervals: 

```@example volume_preserving_attention 
p3 = produce_validation_plot(400)
```

Here the advantage of the attention mechanism with arbitrary weighting vanishes. 