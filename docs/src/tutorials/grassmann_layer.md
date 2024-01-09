# Example of a Neural Network with a Grassmann Layer

Here we show how to implement a neural network that contains a layer whose weight is an element of the Grassmann manifold and where this might be useful. 

To answer where we would need this consider the following scenario

## Problem statement

We are given data in a big space ``\mathcal{D}=[d_i]_{i\in\mathcal{I}}\subset\mathbb{R}^N`` and know these data live on an ``n``-dimensional[^1] submanifold[^2] in ``\mathbb{R}^N``. Based on these data we would now like to generate new samples from the distributions that produced our original data. This is where the Grassmann manifold is useful: each element ``V`` of the Grassmann manifold is an ``n``-dimensional subspace of ``\mathbb{R}^N`` from which we can easily sample. We can then construct a (bijective) mapping from this space ``V`` onto a space that contains our data points ``\mathcal{D}``. 

[^1]: We may know ``n`` exactly or approximately. 
[^2]: Problems and solutions related to this scenario are commonly summarized under the term *manifold learning* (see [lin2008riemannian](@cite)).

## Example

Consider the following toy example: We want to sample from the graph of the Rosenbrock function ``f(x,y) = (1 - x)^2 + 100(y - x^2)^2`` while pretending we do not know the function. 

```@example rosenbrock
using Plots 

rosenbrock(x::Vector) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x, y = -1.5:0.1:1.5, -1.5:0.1:1.5
z = Surface((x,y)->rosenbrock([x,y]), x, y)
p = surface(x,y,z; camera=(30,20), alpha=.6, colorbar=false)
```

We now build a neural network whose task it is to map a product of two Gaussians ``\mathcal{N}(0,1)\times\mathcal{N}(0,1)`` onto the graph of the Rosenbrock function where the range for ``x`` and for ``y`` is ``[-1.5,1.5]``.

```@example rosenbrock
using GeometricMachineLearning, Zygote

model = Chain(GrassmannLayer(2,3), ResNet(3, tanh), ResNet(3, tanh))

nn = NeuralNetwork(model, CPU(), Float64)

function loss(ps::Tuple, nsamples=100)
    samples = randn(2,nsamples)
    estimate = model(samples, ps)
    sum(i -> (estimate[3,i] - rosenbrock(estimate[1:2,i]))^2, 1:nsamples)/nsamples
end

optimizer = Optimizer(nn, AdamOptimizer(Float64))

const training_steps = 10000
loss_array = zeros(training_steps)
for i in 1:training_steps
    val, pullback = Zygote.pullback(ps -> loss(ps), nn.params)
    loss_array[i] = val
    dp = pullback(1)[1]
    optimization_step!(optimizer, model, nn.params, dp)
end
plot(loss_array, xlabel="training step")
```

Now we plot a few points to check how well they match the graph:

```@example rosenbrock
const number_of_points = 10

coordinates = nn(randn(2, number_of_points))
scatter3d!(p, [coordinates[1, :]], [coordinates[2, :]], [coordinates[3, :]], alpha=.5)
```