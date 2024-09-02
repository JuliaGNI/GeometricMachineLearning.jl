# Adjusting the Loss Function

`GeometricMachineLearning` provides a few standard loss functions that are used as defaults [for specific neural networks](@ref "Different Neural Network Losses").

If these standard losses do not satisfy the user's needs, it is very easy to implement custom loss functions. Adding terms to the loss function is standard practice in machine learning to either increase stability [goodfellow2016deep](@cite) or to *inform* the network about physical properties[^1] [raissi2019physics](@cite).

[^1]: Note however that we discourage using so-called [physics-informed neural networks](@ref "A Note on Physics-Informed Neural Networks") as they do not preserve any physical properties but only give a potential improvement on stability in the region where we have training data.

We again consider training a SympNet on the data coming from a harmonic oscillator:

```@example change_loss
using GeometricMachineLearning  # hide
using GeometricIntegrators: integrate, ImplicitMidpoint  # hide
using GeometricProblems.HarmonicOscillator: hodeproblem
import Random # hide
Random.seed!(123) # hide

sol = integrate(hodeproblem(; tspan = 100), ImplicitMidpoint()) 
data = DataLoader(sol; suppress_info = true)

nn = NeuralNetwork(GSympNet(2))

# train the network
o = Optimizer(AdamOptimizer(), nn)
batch = Batch(32)
n_epochs = 30
loss = FeedForwardLoss()
loss_array = o(nn, data, batch, n_epochs, loss; show_progress = false)
print(loss_array[end])
```

And we see that the loss goes down to a very low value. But the user might want to constrain the norm of the network parameters:

```@example change_loss
using LinearAlgebra: norm  # hide

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

const λ = .1
function (loss::CustomLoss)(model::Chain, params::Tuple, input::CT, output::CT) where {
                                                            T,
                                                            AT<:AbstractArray{T, 3}, 
                                                            CT<:@NamedTuple{q::AT, p::AT}
                                                            }
    FeedForwardLoss()(model, params, input, output) + λ * network_parameter_norm(params)
end
nothing # hide
```

And we train the same network with this new loss:

```@example change_loss
loss = CustomLoss()
nn_custom = NeuralNetwork(GSympNet(2))
loss_array = o(nn_custom, data, batch, n_epochs, loss; show_progress = false)
print(loss_array[end])
```

We see that the norm of the parameters is lower:

```@example change_loss
network_parameter_norm(nn_custom.params)
```

We can also compare the solutions of the two networks:

```@setup change_loss
using CairoMakie

function make_fig(; theme = :dark) # hide
textcolor = theme == :dark ? :white : :black # hide
fig = Figure(; backgroundcolor = :transparent)
ax = Axis(fig[1, 1]; backgroundcolor = :transparent, 
    bottomspinecolor = textcolor, 
    topspinecolor = textcolor,
    leftspinecolor = textcolor,
    rightspinecolor = textcolor,
    xtickcolor = textcolor, 
    ytickcolor = textcolor,
    xticklabelcolor = textcolor,
    yticklabelcolor = textcolor)

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

lines!(ax, data.input.q[:], data.input.p[:], label = rich("Training Data"; color = textcolor), linewidth = 3)
lines!(ax, prediction1[1, :], prediction1[2, :], label = rich("FeedForwardLoss"; color = textcolor), linewidth = 3)
lines!(ax, prediction2[1, :], prediction2[2, :], label = rich("CustomLoss"; color = textcolor), linewidth = 3)
axislegend(; position = (.82, .75), backgroundcolor = :transparent) # hide

fig
end # hide
 # hide
save("compare_losses.png", make_fig(; theme = :light); px_per_unit = 1.2) # hide
save("compare_losses_dark.png", make_fig(; theme = :dark); px_per_unit = 1.2) # hide

nothing
```

```@example
Main.include_graphics("compare_losses"; caption = "Here we trained the same network with two different losses. ") # hide
```

Wit the second loss function, for which the norm of the resulting network parameters has lower value, the network still performs well, albeit slightly worse than the network trained with the first loss.