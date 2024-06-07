# Adjusting the Loss Function

`GeometricMachineLearning` provides a few standard loss function that are used as defaults for specific neural networks:
* [`FeedForwardLoss`](@ref)
* [`AutoEncoderLoss`](@ref)
* [`TransformerLoss`](@ref)

If these standard losses do not satisfy the user's needs, it is very easy to implement custom loss functions. We again consider training a SympNet on the data coming from a pendulum:

```@example change_loss
using GeometricMachineLearning
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.HarmonicOscillator: hodeproblem
import Random
Random.seed!(123)

data = integrate(hodeproblem(; tspan = 100), ImplicitMidpoint()) |> DataLoader

nn = NeuralNetwork(GSympNet(2))

o = Optimizer(AdamOptimizer(), nn)

batch = Batch(32)

n_epochs = 30

loss = FeedForwardLoss()

loss_array = o(nn, data, batch, n_epochs, loss)

print(loss_array[end])
```

And we see that the loss goes down to a very low value. But the user might want to constrain the norm of the network parameters:

```@example change_loss
using LinearAlgebra: norm

# norm of parameters for single layer
network_parameter_norm(params::NamedTuple) = sum([norm(params[i]) for i in 1:length(params)])
# norm of parameters for entire network
network_parameter_norm(params) = sum([network_parameter_norm(param) for param in params])

network_parameter_norm(nn.params)
```

We now implement a custom loss such that:

```math
    \mathrm{loss}_\mathcal{NN}^\mathrm{custom}(\mathrm{input}, \mathrm{output}) = \mathrm{loss}_\mathcal{NN}^\mathrm{feedforward} + \lambda \mathrm{norm}(\mathcal{NN}\mathtt{.params}).
```

```@example change_loss
struct CustomLoss <: GeometricMachineLearning.NetworkLoss end

function (loss::CustomLoss)(model::Chain, params::Tuple, input::CT, output::CT) where {
                                                            AT<:AbstractArray, 
                                                            CT<:@NamedTuple{q::AT, p::AT}
                                                            }
    FeedForwardLoss()(model, params, input, output) + .1 * network_parameter_norm(params)
end

loss = CustomLoss()

nn_custom = NeuralNetwork(GSympNet(2))

loss_array = o(nn_custom, data, batch, n_epochs, loss)

print(loss_array[end])
```

And we see that the norm of the parameters is a lot lower:

```@example change_loss
network_parameter_norm(nn_custom.params)
```

We can also compare the solutions of the two networks:

```@example change_loss
using CairoMakie

function make_fig(; theme = :dark, size = (450, 338)) # hide
textcolor = theme == :dark ? :white : :black # hide
fig = Figure(; backgroundcolor = :transparent)
ax = Axis(fig[1, 1]; backgroundcolor = :transparent, 
    bottomspinecolor = textcolor, 
    topspinecolor = textcolor,
    leftspinecolor = textcolor,
    rightspinecolor = textcolor,
    xtickcolor = textcolor, 
    ytickcolor = textcolor)

init_con = [0.5 0.]
n_time_steps = 100
prediction1 = zeros(2, n_time_steps + 1)
prediction2 = zeros(2, n_time_steps + 1)
prediction1[:, 1] = init_con
prediction2[:, 1] = init_con

for i in 2:(n_time_steps + 1)
    prediction1[:, i] = nn(prediction1[:, i - 1])
    prediction2[:, i] = nn_custom(prediction2[:, i - 1])
end

lines!(ax, data.input.q[:], data.input.p[:], label = rich("Training Data"; color = textcolor))
lines!(ax, prediction1[1, :], prediction1[2, :], label = rich("FeedForwardLoss"; color = textcolor))
lines!(ax, prediction2[1, :], prediction2[2, :], label = rich("CustomLoss"; color = textcolor))
text_color = :white # hide
axislegend(; position = (.82, .75), backgroundcolor = :transparent) # hide

fig
end # hide
 # hide
save("compare_losses.png", make_fig(; theme = :light)) # hide
save("compare_losses_dark.png", make_fig(; theme = :dark)) # hide
Main.include_graphics("compare_losses") # hide
```

## Library Functions

```@docs; canonical = false
GeometricMachineLearning.NetworkLoss
GeometricMachineLearning.FeedForwardLoss
GeometricMachineLearning.AutoEncoderLoss
GeometricMachineLearning.TransformerLoss
```