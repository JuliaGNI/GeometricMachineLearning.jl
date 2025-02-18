# Matrix Softmax v Vector Softmax

In this section we compare the [`VectorSoftmax`](@ref) to the [`MatrixSoftmax`](@ref). What is usually meant by *softmax* is the vector softmax, i.e. one that does:

```math
[\mathrm{softmax}(a)]_i = \frac{e^{a_i}}{\sum_{i'=1}^de^{a_i}}. 
```

So each column of a matrix is normalized to sum up to one. With this softmax, the linear recombination that is performed [by the attention layer](@ref "Multihead Attention") becomes a *convex recombination*. This is not the case for the [`MatrixSoftmax`](@ref), where the normalization is computed over all matrix entries:

```math
[\mathrm{softmax}(A)]_{ij} = \frac{e^{A_{ij}}}{\sum_{i'=1, j'=1}^{d,\bar{d}}e^{A_{ij}}}. 
```

We want to compare those two approaches on the example of the *coupled harmonic oscillator*. It is a [Hamiltonian system](@ref "Symplectic Systems") with 

```math
H(q_1, q_2, p_1, p_2) = \frac{q_1^2}{2m_1} + \frac{q_2^2}{2m_2} + k_1\frac{q_1^2}{2} + k_2\frac{q_2^2}{2} +  k\sigma(q_1)\frac{(q_2 - q_1)^2}{2},
```
where ``\sigma(x) = 1 / (1 + e^{-x})`` is the sigmoid activation function. The system parameters are:
- ``k_1``: spring constant belonging to ``m_1``,
- ``k_2``: spring constant belonging to ``m_2``,
- ``m_1``: mass 1,
- ``m_2``: mass 2,
- ``k``: coupling strength between the two masses. 

![Visualization of the coupled harmonic oscillator.](../tikz/coupled_harmonic_oscillator_light.png)
![Visualization of the coupled harmonic oscillator.](../tikz/coupled_harmonic_oscillator_dark.png)

We will leave the parameters fixed but alter the initial conditions[^1]:

[^1]: We here use the implementation of the coupled harmonic oscillator from [`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl).

```@example softmax_comparison
using GeometricMachineLearning # hide
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate # hide
using LaTeXStrings # hide
using CairoMakie  # hide
CairoMakie.activate!() # hide
import Random # hide
Random.seed!(123) # hide

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256) # hide
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256) # hide

const tstep = .3
const n_init_con = 5

# ensemble problem
ep = hodeensemble([rand(2) for _ in 1:n_init_con], [rand(2) for _ in 1:n_init_con]; tstep = tstep)
dl = DataLoader(integrate(ep, ImplicitMidpoint()); suppress_info = true)
# dl = DataLoader(vcat(dl_nt.input.q, dl_nt.input.p))  # hide

nothing # hide
```

We now use the same architecture, a [`TransformerIntegrator`](@ref), twice, but alter its activation function:

```@example softmax_comparison
const seq_length = 4
const batch_size = 1024
const n_epochs = 1000

act1 = GeometricMachineLearning.VectorSoftmax()
act2 = GeometricMachineLearning.MatrixSoftmax()

arch1 = StandardTransformerIntegrator(dl.input_dim; transformer_dim = 20,
                                                    n_heads = 4, 
                                                    L = 1, 
                                                    n_blocks = 2,
                                                    attention_activation = act1)

arch2 = StandardTransformerIntegrator(dl.input_dim; transformer_dim = 20,
                                                    n_heads = 4,
                                                    L = 1,
                                                    n_blocks = 2,
                                                    attention_activation = act2)

nn1 = NeuralNetwork(arch1)
nn2 = NeuralNetwork(arch2)
```

Training is done with the [`AdamOptimizer`](@ref):

```@example softmax_comparison
o_method = AdamOptimizer()

o1 = Optimizer(o_method, nn1)
o2 = Optimizer(o_method, nn2)

batch = Batch(batch_size, seq_length)

loss_array1 = o1(nn1, dl, batch, n_epochs)
loss_array2 = o2(nn2, dl, batch, n_epochs)
```

```@setup softmax_comparison
function make_training_error_plot(; theme = :dark)
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

    # we use linewidth  = 2
    lines!(ax, loss_array1; color = mpurple, label = "VecSoftM", linewidth = 2)
    lines!(ax, loss_array2; color = mred, label = "MatSoftM", linewidth = 2)
    axislegend(; position = (.55, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

training_fig_light, training_ax_light = make_training_error_plot(; theme = :light)
training_fig_dark, training_ax_dark = make_training_error_plot(; theme = :dark)
save("attention_training_dark.png", training_fig_dark; px_per_unit = 1.2)
save("attention_training_light.png", training_fig_light; px_per_unit = 1.2)

nothing
```

![Training loss for the different networks.](attention_training_light.png)
![Training loss for the different networks.](attention_training_dark.png)

```@setup softmax_comparison
const index = 1
init_con = (q = dl.input.q[:, 1:seq_length, index], p = dl.input.p[:, 1:seq_length, index])

const n_steps = 300

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
    prediction_vector = iterate(nn1, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_matrix = iterate(nn2, init_con; n_points = n_steps, prediction_window = seq_length)

    # we use linewidth  = 2
    lines!(ax, dl.input.q[1, 1:n_steps, index]; color = mblue, label = "Implicit midpoint", linewidth = 2)
    lines!(ax, prediction_vector.q[1, :]; color = mpurple, label = "VecSoftM", linewidth = 2)
    lines!(ax, prediction_matrix.q[1, :]; color = mred, label = "MatSoftM", linewidth = 2)
    axislegend(; position = (.55, .75), backgroundcolor = :transparent, labelcolor = textcolor)

    fig, ax
end

fig_light, ax_light = make_validation_plot(n_steps; theme = :light)
fig_dark, ax_dark = make_validation_plot(n_steps; theme = :dark)
save("validation_dark.png", fig_dark; px_per_unit = 1.2)
save("validation_light.png", fig_light; px_per_unit = 1.2)

nothing
```

![Predicting trajectories with transformers based on the vector softmax and the matrix softmax.](validation_light.png)
![Predicting trajectories with transformers based on the vector softmax and the matrix softmax.](validation_dark.png)

A similar page can be found [here](@ref "Comparing Matrix and Vector Softmax as Activation Functions in a Transformer").