# [Linear Symplectic Transformer](@id linear_symplectic_transformer_tutorial)

In this section we compare the [linear symplectic transformer](@ref "Linear Symplectic Transformer") to the [standard transformer](@ref "Standard Transformer"). The example we treat here is the *coupled harmonic oscillator*:

```@example
Main.include_graphics("../tikz/coupled_harmonic_oscillator"; caption = raw"Visualization of the coupled harmonic oscillator.") # hide
```

It is a [Hamiltonian system](@ref "Symplectic Systems") with 

```math
H(q_1, q_2, p_1, p_2) = \frac{q_1^2}{2m_1} + \frac{q_2^2}{2m_2} + k_1\frac{q_1^2}{2} + k_2\frac{q_2^2}{2} +  k\sigma(q_1)\frac{(q_2 - q_1)^2}{2},
```
where ``\sigma(x) = 1 / (1 + e^{-x})`` is the sigmoid activation function. The system parameters are:
- ``k_1``: spring constant belonging to ``m_1``,
- ``k_2``: spring constant belonging to ``m_2``,
- ``m_1``: mass 1,
- ``m_2``: mass 2,
- ``k``: coupling strength between the two masses. 

To demonstrate the efficacy of the linear symplectic transformer here we will leave the parameters fixed but alter the initial conditions[^1]:

[^1]: We here use the implementation of the coupled harmonic oscillator from [`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl).

```@example lin_sympl_tran_tut
using GeometricMachineLearning # hide
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate # hide
using LaTeXStrings # hide
using CairoMakie  # hide
CairoMakie.activate!() # hide
import Random # hide
Random.seed!(123) # hide

const tstep = .3
const n_init_con = 5

# ensemble problem
ep = hodeensemble([rand(2) for _ in 1:n_init_con], [rand(2) for _ in 1:n_init_con]; tstep = tstep)
dl = DataLoader(integrate(ep, ImplicitMidpoint()); suppress_info = true)
# dl = DataLoader(vcat(dl_nt.input.q, dl_nt.input.p))  # hide

nothing # hide
```

We now define the architectures and train them: 

```@example lin_sympl_tran_tut
const seq_length = 4
const batch_size = 1024
const n_epochs = 2000

arch_standard = StandardTransformerIntegrator(dl.input_dim; n_heads = 2, 
                                                            L = 1, 
                                                            n_blocks = 2)
arch_symplectic = LinearSymplecticTransformer(  dl.input_dim, 
                                                seq_length; n_sympnet = 2,
                                                L = 1, 
                                                upscaling_dimension = 2 * dl.input_dim)
arch_sympnet = GSympNet(dl.input_dim;   n_layers = 4, 
                                        upscaling_dimension = 2 * dl.input_dim)

nn_standard = NeuralNetwork(arch_standard)
nn_symplectic = NeuralNetwork(arch_symplectic)
nn_sympnet = NeuralNetwork(arch_sympnet)

o_method = AdamOptimizerWithDecay(n_epochs, Float64)

o_standard = Optimizer(o_method, nn_standard)
o_symplectic = Optimizer(o_method, nn_symplectic)
o_sympnet = Optimizer(o_method, nn_sympnet)

batch = Batch(batch_size, seq_length)
batch2 = Batch(batch_size)

loss_array_standard = o_standard(nn_standard, dl, batch, n_epochs; show_progress = false)
loss_array_symplectic = o_symplectic(nn_symplectic, dl, batch, n_epochs; show_progress = false)
loss_array_sympnet = o_sympnet(nn_sympnet, dl, batch2, n_epochs; show_progress = false)

nothing # hide
```

And the corresponding training losses look as follows:

```@setup lin_sympl_tran_tut
morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256) # hide
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256) # hide

function plot_training_losses(loss_array_standard, loss_array_symplectic, loss_array_sympnet; theme = :dark)
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
    lines!(ax, loss_array_standard, color = mpurple, label = "ST")
    lines!(ax, loss_array_symplectic,  color = mred, label = "LST")
    lines!(ax, loss_array_sympnet, color = morange, label = "SympNet")
    axislegend(; position = (.82, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

fig_dark, ax_dark = plot_training_losses(loss_array_standard, loss_array_symplectic, loss_array_sympnet; theme = :dark)
fig_light, ax_light = plot_training_losses(loss_array_standard, loss_array_symplectic, loss_array_sympnet; theme = :light)

save("lst_dark.png", fig_dark; px_per_unit = 1.2)
save("lst.png", fig_light; px_per_unit = 1.2)

nothing
```

```@example
Main.include_graphics("lst"; caption = "Training loss for the different networks.") # hide
```


We further evaluate a trajectory with the trained networks for thirty time steps: 

```@setup lin_sympl_tran_tut
const index = 1
init_con = (q = dl.input.q[:, 1:seq_length, index], p = dl.input.p[:, 1:seq_length, index])
# when we iterate with a feedforward neural network we only need a vector as input
init_con_ff = (q = dl.input.q[:, 1, index], p = dl.input.p[:, 1, index])

const n_steps = 30

function make_validation_plot(n_steps = n_steps; theme = :dark)
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
        xlabel=L"t", 
        ylabel=L"q_1",
        xlabelcolor = textcolor,
        ylabelcolor = textcolor,
    )
    prediction_standard = iterate(nn_standard, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_symplectic = iterate(nn_symplectic, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_sympnet = iterate(nn_sympnet, init_con_ff; n_points = n_steps)

    # we use linewidth  = 2
    lines!(ax, dl.input.q[1, 1:n_steps, index]; color = mblue, label = "Implicit midpoint", linewidth = 2)
    lines!(ax, prediction_standard.q[1, :]; color = mpurple, label = "ST", linewidth = 2)
    lines!(ax, prediction_symplectic.q[1, :]; color = mred, label = "LST", linewidth = 2)
    lines!(ax, prediction_sympnet.q[1, :]; color = morange, label = "SympNet", linewidth = 2)
    axislegend(; position = (.55, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

fig_light, ax_light = make_validation_plot(n_steps; theme = :light)
fig_dark, ax_dark = make_validation_plot(n_steps; theme = :dark)
save("lst_validation.png", fig_light; px_per_unit = 1.2)
save("lst_validation_dark.png", fig_dark; px_per_unit = 1.2)

nothing
```

```@example lin_sympl_tran_tut
Main.include_graphics("lst_validation"; caption = "Validation of the different networks.", width = .85) # hide
```

We can see that the standard transformer is not able to stay close to the trajectory coming from implicit midpoint very well. The linear symplectic transformer outperforms the standard transformer as well as the SympNet while needing fewer parameters than the standard transformer: 

```@example lin_sympl_tran_tut
parameterlength(nn_standard), parameterlength(nn_symplectic), parameterlength(nn_sympnet)
```

It is also interesting to note that the training error for the SympNet gets lower than the one for the linear symplectic transformer, but it does not manage to outperform it when looking at the validation. 