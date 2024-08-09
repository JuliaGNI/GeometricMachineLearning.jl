# Example of a Neural Network with a Grassmann Layer

Here we show how to implement a neural network that contains a layer whose weight is an element of the [Grassmann manifold](@ref "The Grassmann Manifold") and where this might be useful. Recall that the Grassmann manifold ``Gr(n, N)`` is the set of vector spaces of dimension ``n`` embedded in ``\mathbb{R}^N``. So if we optimize on the Grassmann manifold, we optimize for an *ideal* ``n``-dimensional vector space in the bigger space ``\mathbb{R}^N``. 

```@eval
Main.remark(raw"This example for learning weights on a Grassmann manifold also shows how `GeometricMachineLearning` can be used together with other packages. Here we use the *Wasserstein distance* from the package `BrenierTwoFluid` for example.")
```

## Identifying Non-Linear Subspaces of ``\mathbb{R}^N``

Consider the following scenario: we are given data in a big space ``\mathcal{D}=[d_i]_{i\in\mathcal{I}}\subset\mathbb{R}^N`` and know these data live on an ``n``-dimensional submanifold[^1] in ``\mathbb{R}^N``. Based on these data we would now like to generate new samples from the distributions that produced our original data. This is where the Grassmann manifold is useful: each element ``V`` of the Grassmann manifold is an ``n``-dimensional subspace of ``\mathbb{R}^N`` from which we can easily sample. We can then construct a (bijective) mapping from this space ``V`` onto a space that contains our data points ``\mathcal{D}``. 

[^1]: Problems and solutions related to this scenario are part of the *manifold learning paradigm* (see [lin2008riemannian](@cite)). [Data-driven reduced order modeling](@ref "General Workflow") can also be seen as belonging to this category.

## Academic Example

Consider the following toy example: We want to sample from the graph of the (scaled) Rosenbrock function 

```math
f(x,y) = ((1 - x)^2 + 100(y - x^2)^2)/1000
``` 
without using the explicit form of the function during sampling. We show the graph of ``f`` for ``(x, y)\in[-1.5, 1.5]^2`` in the following picture:

```@setup rosenbrock
using GLMakie, LaTeXStrings
GLMakie.activate!() # hide
include("../../gl_makie_transparent_background_hack.jl")

rosenbrock(x::Vector) = ((1.0 - x[1]) ^ 2 + 100.0 * (x[2] - x[1] ^ 2) ^ 2) / 1000
x, y = -1.5:0.1:1.5, -1.5:0.1:1.5
z = [rosenbrock([x,y]) for x in x, y in y]
function make_rosenbrock(; theme = :dark, alpha = .7) # hide
textcolor = theme == :dark ? :white : :black
fig = Figure(; backgroundcolor = :transparent, size = (900, 675))
ax = Axis3(fig[1, 1];
                     limits = ((-1.5, 1.5), (-1.5, 1.5), (0.0, rosenbrock([-1.5, -1.5]))),
                     azimuth = π / 6,
                     elevation = π / 8,
                     backgroundcolor = (:tomato, .5), # hide
                     xgridcolor = textcolor, 
                     ygridcolor = textcolor, 
                     zgridcolor = textcolor,
                     xtickcolor = textcolor, 
                     ytickcolor = textcolor,
                     ztickcolor = textcolor,
                     xticklabelcolor = textcolor,
                     yticklabelcolor = textcolor,
                     zticklabelcolor = textcolor,
                     xypanelcolor = :transparent,
                     xzpanelcolor = :transparent,
                     yzpanelcolor = :transparent,
                     xlabel = L"x", 
                     ylabel = L"y",
                     zlabel = L"z",
                     xlabelcolor = textcolor,
                     ylabelcolor = textcolor,
                     zlabelcolor = textcolor)
surface!(ax, x, y, z; alpha = alpha, transparency = true)

fig, ax
end # hide

fig_dark, ax_dark = make_rosenbrock(; theme = :dark, alpha = .85)
fig_light, ax_light = make_rosenbrock(; theme = :light)
save("rosenbrock_dark.png", alpha_colorbuffer(fig_dark))
save("rosenbrock.png", alpha_colorbuffer(fig_light))

nothing # hide
```

```@example rosenbrock
Main.include_graphics("rosenbrock") # hide
```

We now build a neural network whose task it is to map a product of two Gaussians ``\mathcal{N}(0,1)\times\mathcal{N}(0,1)`` onto the graph of the Rosenbrock function:

```math
    \mathcal{NN}: \mathcal{N}(0,1)\times\mathcal{N}(0,1) \to \{(x, y, z): (x, y)\in[-1.5, 1.5]\times[-1.5, 1.5], z = f(x, y)\},
```
where ``f`` is the Rosenbrock function. 

For computing the loss between the two distributions, i.e. ``\Psi(\mathcal{N}(0,1)\times\mathcal{N}(0,1))`` and ``f([-1.5,1.5], [-1.5,1.5])`` we use the Wasserstein distance[^2].

[^2]: The implementation of the Wasserstein distance is taken from [blickhan2023brenier](@cite).

We first set up the neural network. It consists of three layers:

```@example rosenbrock
using GeometricMachineLearning, Zygote, BrenierTwoFluid
using LinearAlgebra: norm # hide
import Random # hide 
Random.seed!(1234) # hide

model = Chain(GrassmannLayer(2,3), Dense(3, 8, tanh), Dense(8, 3, identity))

nn = NeuralNetwork(model, CPU(), Float64)

nothing # hide
```

We then *lift* the neural network parameters via [`GlobalSection`](@ref).

```@example rosenbrock
λY = GlobalSection(nn.params)

nothing # hide
```

As the cost function ``c`` for the Wasserstein loss[^3] we simply use:

[^3]: This means that we have a ``W_2`` loss here.

```@example rosenbrock
# this computes the cost that is associated to the Wasserstein distance
c = (x,y) -> .5 * norm(x - y)^2
∇c = (x,y) -> x - y

nothing # hide
```

We then define a function `compute_wasserstein_gradient`:

```@example rosenbrock
const ε = 0.1                 # entropic regularization. √ε is a length.  # hide
const q = 1.0                 # annealing parameter                       # hide
const Δ = 1.0                 # characteristic domain size                # hide
const s = ε                   # current scale: no annealing -> equals ε   # hide
const tol = 1e-6              # marginal condition tolerance              # hide
const crit_it = 20            # acceleration inference                    # hide
const p_η = 2                                                             # hide

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

nothing # hide
```

This function associates particles in two point clouds with each other. As an illustrative example we will compare the following two point clouds: 

```math
    \mathcal{D}_1 = \{(x, y, z): (x, y) \in \mathtt{-1.5:0.1:1.5}^2, z = f(x, y, z)\}
```

and

```math
    \mathcal{D}_2 = \left\{  \begin{pmatrix} 2 \\ 2 \\ 2 \end{pmatrix} + x: x \sim \mathtt{rand} \right\},
```

where ``x \sim \mathtt{rand}`` means that we draw ``x`` with the function `rand`. In code the two sets are:

```@example rosenbrock
xyz_points = hcat([[x, y, rosenbrock([x,y])] for x in x for y in y]...)
nothing # hide
```
and 

```@example rosenbrock
point_cloud = rand(size(xyz_points)...) .+ [2., 2., 2.]
nothing # hide
```

We then compute the Wasserstein gradients and plot 30 of those (picked at random):

```@setup rosenbrock
morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256) # hide

loss, grads = compute_wasserstein_gradient(point_cloud, xyz_points)

function make_point_cloud_arrows(; theme = :dark)
textcolor = theme == :dark ? :white : :black
fig = Figure(; backgroundcolor = :transparent, size = (900, 675))
ax = Axis3(fig[1, 1];
                     limits = ((-1.5, 1.5), (-1.5, 1.5), (0.0, rosenbrock([-1.5, -1.5]))),
                     azimuth = π / 6,
                     elevation = π / 8,
                     backgroundcolor = (:tomato, .5), # hide
                     xgridcolor = textcolor, 
                     ygridcolor = textcolor, 
                     zgridcolor = textcolor,
                     xtickcolor = textcolor, 
                     ytickcolor = textcolor,
                     ztickcolor = textcolor,
                     xticklabelcolor = textcolor,
                     yticklabelcolor = textcolor,
                     zticklabelcolor = textcolor,
                     xypanelcolor = :transparent,
                     xzpanelcolor = :transparent,
                     yzpanelcolor = :transparent,
                     xlabel = L"x", 
                     ylabel = L"y",
                     zlabel = L"z",
                     xlabelcolor = textcolor,
                     ylabelcolor = textcolor,
                     zlabelcolor = textcolor)

scatter!(ax, point_cloud; color = mred, alpha = .6, label = L"\mathcal{D}_1")
scatter!(ax, xyz_points; color = mblue, alpha = .6, label = L"\mathcal{D}_2")

number_arrows_drawn = 30
indices = Int.(ceil.(size(point_cloud, 2) * rand(number_arrows_drawn)))
arrows!(ax, point_cloud[1, indices], point_cloud[2, indices], point_cloud[3, indices], 
            - grads[1, indices],     - grads[2, indices],     - grads[3, indices]; 
            color = mred, 
            linewidth = .01, 
            alpha = .01, 
            arrowsize = .04,
            transparency = true
            )
axislegend(; position = (.92, .75), backgroundcolor = :transparent, labelcolor = textcolor) # hide
fig, ax
end
fig_light, ax_light = make_point_cloud_arrows(; theme = :light)
fig_dark, ax_dark = make_point_cloud_arrows(; theme = :dark)

save("point_cloud_arrows.png", alpha_colorbuffer(fig_light))
save("point_cloud_arrows_dark.png", alpha_colorbuffer(fig_dark))

nothing # hide
```

```@example rosenbrock
Main.include_graphics("point_cloud_arrows") # hide
```

We now want to train a neural network based on this Wasserstein loss. The loss function is:

```math
L_\mathcal{NN}(\theta) = W_2(\mathcal{NN}_\theta([x^{(1)}, \ldots, x^{(\mathtt{np})}]), \mathcal{D}_2),
```
where `np` is the number of points in ``\mathcal{D}_2.`` We then have

```math
\nabla_\theta{}L_\mathcal{NN} = (\nabla{}W_2)\nabla_\theta\mathcal{NN},
```
where ``\nabla{}W_2`` is equivalent to the function `compute_wasserstein_gradient`.

```@example rosenbrock
xyz_points = hcat([[x, y, rosenbrock([x,y])] for x in x for y in y]...)

function compute_gradient(ps::Tuple)
    samples = randn(2, size(xyz_points, 2))

    estimate, nn_pullback = Zygote.pullback(ps -> model(samples, ps), ps)

    valS, wasserstein_gradient = compute_wasserstein_gradient(estimate, xyz_points)
    valS, nn_pullback(wasserstein_gradient)[1]
end

# note the very high value for the learning rate
optimizer = Optimizer(nn, AdamOptimizer(1e-1))

nothing # hide
```

We can now train our network:

```@example rosenbrock
import CairoMakie # hide
CairoMakie.activate!() # hide
# note the small number of training steps
const training_steps = 80
loss_array = zeros(training_steps)
for i in 1:training_steps
    val, dp = compute_gradient(nn.params)
    loss_array[i] = val
    optimization_step!(optimizer, λY, nn.params, dp)
end
```

and plot the training loss:
```@setup rosenbrock
function make_error_plot(; theme = :dark) # hide
textcolor = theme == :dark ? :white : :black # hide
fig = CairoMakie.Figure(; backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1]; 
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
    ylabel="Training error",
    xlabelcolor = textcolor,
    ylabelcolor = textcolor,
    yscale = log10
    )

CairoMakie.lines!(ax, loss_array, label = "training loss", color = mblue)
CairoMakie.axislegend(; position = (.82, .75), backgroundcolor = :transparent, labelcolor = textcolor) # hide
fig_name = theme == :dark ? "training_loss_dark.png" : "training_loss.png" # hide
CairoMakie.save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_error_plot(; theme = :dark) # hide
make_error_plot(; theme = :light) # hide

nothing # hide
```

```@example rosenbrock
Main.include_graphics("training_loss") # hide
```

Now we plot a few points to check how well they match the graph:

```@example rosenbrock
using GLMakie # hide
GLMakie.activate!() # hide

Random.seed!(124)  # hide
const number_of_points = 35

coordinates = nn(randn(2, number_of_points))

for theme in (:dark, :light) # hide
fig, ax = make_rosenbrock(; theme = theme) # hide
scatter!(ax, coordinates[1, :], coordinates[2, :], coordinates[3, :]; 
            alpha = .9, 
            color = mblue, 
            label="mapped points")
textcolor = theme == :dark ? :white : :black # hide
axislegend(; position = (.82, .75), backgroundcolor = :transparent, labelcolor = textcolor) # hide
file_name = "mapped_points" * (theme == :dark ? "_dark.png" : ".png") # hide
save(file_name, alpha_colorbuffer(fig)) # hide
end # hide

Main.include_graphics("mapped_points") # hide
```

If points appear in darker color this means that they lie behind the graph of the Rosenbrock function.

## Library Functions

```@docs
GrassmannLayer
```