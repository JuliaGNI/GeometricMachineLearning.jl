# Example of a Neural Network with a Grassmann Layer

Here we show how to implement a neural network that contains a layer whose weight is an element of the Grassmann manifold and where this might be useful. 

To answer where we would need this consider the following scenario

## Problem statement

We are given data in a big space ``\mathcal{D}=[d_i]_{i\in\mathcal{I}}\subset\mathbb{R}^N`` and know these data live on an ``n``-dimensional[^1] submanifold[^2] in ``\mathbb{R}^N``. Based on these data we would now like to generate new samples from the distributions that produced our original data. This is where the Grassmann manifold is useful: each element ``V`` of the Grassmann manifold is an ``n``-dimensional subspace of ``\mathbb{R}^N`` from which we can easily sample. We can then construct a (bijective) mapping from this space ``V`` onto a space that contains our data points ``\mathcal{D}``. 

[^1]: We may know ``n`` exactly or approximately. 
[^2]: Problems and solutions related to this scenario are commonly summarized under the term *manifold learning* (see [lin2008riemannian](@cite)).

## Example

Consider the following toy example: We want to sample from the graph of the (scaled) Rosenbrock function ``f(x,y) = ((1 - x)^2 + 100(y - x^2)^2)/1000`` while pretending we do not know the function. 

```@example rosenbrock
using Plots # hide
# hide
rosenbrock(x::Vector) = ((1.0 - x[1]) ^ 2 + 100.0 * (x[2] - x[1] ^ 2) ^ 2) / 1000
x, y = -1.5:0.1:1.5, -1.5:0.1:1.5
z = Surface((x,y)->rosenbrock([x,y]), x, y)
p = surface(x,y,z; camera=(30,20), alpha=.6, colorbar=false, xlims=(-1.5, 1.5), ylims=(-1.5, 1.5), zlims=(0.0, rosenbrock([-1.5, -1.5])))
```

We now build a neural network whose task it is to map a product of two Gaussians ``\mathcal{N}(0,1)\times\mathcal{N}(0,1)`` onto the graph of the Rosenbrock function where the range for ``x`` and for ``y`` is ``[-1.5,1.5]``.

For computing the loss between the two distributions, i.e. ``\Psi(\mathcal{N}(0,1)\times\mathcal{N}(0,1))`` and ``f([-1.5,1.5], [-1.5,1.5])`` we use the Wasserstein distance[^3].

[^3]: The implementation of the Wasserstein distance is taken from [blickhan2023brenier](@cite).

```@example rosenbrock
using GeometricMachineLearning, Zygote, BrenierTwoFluid
using LinearAlgebra: norm # hide
import Random # hide 
Random.seed!(123)

model = Chain(GrassmannLayer(2,3), Dense(3, 8, tanh), Dense(8, 3, identity))

nn = NeuralNetwork(model, CPU(), Float64)

# this computes the cost that is associated to the Wasserstein distance
c = (x,y) -> .5 * norm(x - y)^2
∇c = (x,y) -> x - y

const ε = 0.1                 # entropic regularization. √ε is a length.  # hide
const q = 1.0                 # annealing parameter                       # hide
const Δ = 1.0                 # characteristic domain size                # hide
const s = ε                   # current scale: no annealing -> equals ε   # hide
const tol = 1e-6              # marginal condition tolerance              # hide 
const crit_it = 20            # acceleration inference                    # hide
const p_η = 2

function compute_wasserstein_gradient(ensemble1::AT, ensemble2::AT) where AT<:AbstractArray
    number_of_particles1 = size(ensemble1, 2)
    number_of_particles2 = size(ensemble2, 2)
    V = SinkhornVariable(copy(ensemble1'), ones(number_of_particles1) / number_of_particles1)
    W = SinkhornVariable(copy(ensemble2'), ones(number_of_particles2) / number_of_particles2)
    params = SinkhornParameters(; ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_η=p_η,sym=false,acc=true) # hide
    S = SinkhornDivergence(V, W, c, params; islog = true)
    initialize_potentials!(S)
    compute!(S)
    value(S), x_gradient!(S, ∇c)'
end

xyz_points = hcat([[x,y,rosenbrock([x,y])] for x in x for y in y]...)

function compute_gradient(ps::Tuple)
    samples = randn(2, size(xyz_points, 2))

    estimate, nn_pullback = Zygote.pullback(ps -> model(samples, ps), ps)

    valS, wasserstein_gradient = compute_wasserstein_gradient(estimate, xyz_points)
    valS, nn_pullback(wasserstein_gradient)[1]
end

# note the very high value for the learning rate
optimizer = Optimizer(nn, AdamOptimizer(1e-1))

# note the small number of training steps
const training_steps = 40
loss_array = zeros(training_steps)
for i in 1:training_steps
    val, dp = compute_gradient(nn.params)
    loss_array[i] = val
    optimization_step!(optimizer, model, nn.params, dp)
end
plot(loss_array, xlabel="training step", label="loss")
```

Now we plot a few points to check how well they match the graph:

```@example rosenbrock
const number_of_points = 35

coordinates = nn(randn(2, number_of_points))
scatter3d!(p, [coordinates[1, :]], [coordinates[2, :]], [coordinates[3, :]], alpha=.5, color=4, label="mapped points")
```