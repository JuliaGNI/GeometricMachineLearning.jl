```@raw latex
In this chapter we show how to use symplectic autoencoders for the offline phase of reduced order modeling. The example we consider here is the \textit{Toda lattice}. We further discuss the online phase and compare a standard integrator (implicit midpoint) with a transformer neural network.
```

# Symplectic Autoencoders and the Toda Lattice

In this tutorial we use a [symplectic autoencoder](@ref "The Symplectic Autoencoder") to approximate the solution of the Toda lattice with a lower-dimensional Hamiltonian model and compare it with standard [proper symplectic decomposition](@ref "Proper Symplectic Decomposition").

```@eval
Main.remark(raw"As with any neural network we have to make the following choices:
" * Main.indentation * raw"1. specify the *architecture*,
" * Main.indentation * raw"2. specify the *type* and *backend*,
" * Main.indentation * raw"3. pick an *optimizer* for training the network,
" * Main.indentation * raw"4. specify how you want to perform *batching*,
" * Main.indentation * raw"5. choose a *number of epochs*,
" * Main.indentation * raw"where points 1 and 3 depend on a variable number of hyperparameters.")
```
For the symplectic autoencoder point 1 is done by calling [`SymplecticAutoencoder`](@ref), point 2 is done by calling `NeuralNetwork`, point 3 is done by calling [`Optimizer`](@ref) and point 4 is done by calling [`Batch`](@ref).

## The system

The Toda lattice [toda1967vibration](@cite) is a prototypical example of a Hamiltonian PDE. It is described by 
```math
    H(q, p) = \sum_{n\in\mathbb{Z}}\left(  \frac{p_n^2}{2} + \alpha e^{q_n - q_{n+1}} \right).
```

Starting from this equation we further assume a finite number of particles ``N`` and impose periodic boundary conditions: 
```math
\begin{aligned}
    q_{n+N} &  \equiv q_n \\ 
    p_{n+N} &   \equiv p_n.
\end{aligned}
```

In this tutorial we want to reduce the dimension of the big system by a significant factor with (i) proper symplectic decomposition (PSD) and (ii) symplectic autoencoders (SAE). The first approach is strictly linear whereas the second one allows for more general mappings. 

### Using the Toda lattice in numerical experiments 

In order to use the Toda lattice in numerical experiments we have to pick suitable initial conditions. For this, consider the third-degree spline: 

```math
h(s)  = \begin{cases}
        1 - \frac{3}{2}s^2 + \frac{3}{4}s^3 & \text{if } 0 \leq s \leq 1 \\ 
        \frac{1}{4}(2 - s)^3 & \text{if } 1 < s \leq 2 \\ 
        0 & \text{else.} 
\end{cases}
```

Plotted on the relevant domain it looks like this: 

```@example 
Main.include_graphics("../tikz/third_degree_spline") # hide
```


We end up with the following choice of parametrized initial conditions: 

```math 
u_0(\mu)(\omega) = h(s(\omega, \mu)), \quad s(\omega, \mu) =  20 \mu  |\omega + \frac{\mu}{2}|,
```

where the ``\omega`` is an element of the domain ``\Omega = [-0.5, 0.5].`` For the purposes of this tutorial we will use the default value for ``\mu`` provided in [`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl):

```@example toda_lattice
import GeometricProblems.TodaLattice as tl

N = tl.Ñ # hide
Δx = 1. / (N - 1) # hide
Ω = -0.5 : Δx : 0.5 # hide
tl.μ
```

We thus look at the displacement of ``N = 200`` particles on the periodic domain ``\Omega = [-0.5, 0.5]/~ \simeq S^1`` where the equivalence relation ``~`` indicates that we associate the points ``-0.5`` and ``0.5`` with each other.

## Get the data 

The training data can very easily be obtained by using the packages [`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl) and [`GeometricIntegrators`](https://github.com/JuliaGNI/GeometricIntegrators.jl):

```@example toda_lattice
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricMachineLearning 
import Random # hide

pr = tl.hodeproblem(; tspan = (0.0, 800.))
sol = integrate(pr, ImplicitMidpoint())
nothing # hide
```

We then put the format in the correct format by calling [`DataLoader`](@ref)[^1]:

[^1]: For more information on [`DataLoader`](@ref) see the [corresponding section](@ref "The Data Loader").

```@example toda_lattice
dl_cpu = DataLoader(sol; autoencoder = true, suppress_info = true)
nothing # hide
```

Also note that the keyword `autoencoder` was set to true when calling [`DataLoader`](@ref). The keyword argument `supress_info` determines whether data loader provides some additional information on the data it is called on.

## Train the network 

We now want to compare two different approaches: [`PSDArch`](@ref) and [`SymplecticAutoencoder`](@ref). For this we first have to set up the networks: 

```@example toda_lattice
const reduced_dim = 2

Random.seed!(123) # hide
psd_arch = PSDArch(dl_cpu.input_dim, reduced_dim)
sae_arch = SymplecticAutoencoder(dl_cpu.input_dim, reduced_dim; n_encoder_blocks = 4, 
                                                                n_decoder_blocks = 4, 
                                                                n_encoder_layers = 2, 
                                                                n_decoder_layers = 2)
nothing # hide
```

Training a neural network is usually done by calling an instance of [`Optimizer`](@ref) in `GeometricMachineLearning`. [`PSDArch`](@ref) however can be solved directly by using singular value decomposition and this is done by calling [solve!](@ref):  

```@example toda_lattice
psd_nn_cpu = NeuralNetwork(psd_arch, CPU(), eltype(dl_cpu))

solve!(psd_nn_cpu, dl_cpu)
```

The `SymplecticAutoencoder` we train with the [`AdamOptimizerWithDecay`](@ref) however[^2]:

[^2]: It is not feasible to perform the training on CPU, which is why we use `CUDA` [besard2018juliagpu](@cite) here. We further perform the training in single precision.

```julia
using CUDA

const n_epochs = 262144
const batch_size = 4096

backend = CUDABackend()
dl = DataLoader(dl_cpu, backend, Float32)


sae_nn_gpu = NeuralNetwork(sae_arch, CUDADevice(), Float32)
o = Optimizer(sae_nn_gpu, AdamOptimizerWithDecay(integrator_train_epochs))

# train the network
o(sae_nn_gpu, dl, Batch(batch_size), n_epochs)
```

After training we map the network parameters to cpu:

```@example toda_lattice
const mtc = GeometricMachineLearning.map_to_cpu
nothing # hide
```
```julia
sae_nn_cpu = mtc(sae_nn_gpu)
```

```@setup toda_lattice
using JLD2

sae_trained_parameters = load("sae_parameters.jld2")["sae_parameters"]
_nnp(ps::Tuple) = NeuralNetworkParameters{Tuple(Symbol("L$(i)") for i in 1:length(ps))}(ps)
sae_nn_cpu = NeuralNetwork(sae_arch, Chain(sae_arch), _nnp(sae_trained_parameters), CPU())

nothing  # hide
```

## The online stage with a standard integrator

After having trained our neural network we can now evaluate it in the online stage of reduced complexity modeling: 

```@example toda_lattice
psd_rs = HRedSys(pr, encoder(psd_nn_cpu), decoder(psd_nn_cpu); integrator = ImplicitMidpoint())
sae_rs = HRedSys(pr, encoder(sae_nn_cpu), decoder(sae_nn_cpu); integrator = ImplicitMidpoint())

nothing  # hide
```

We integrate the full system (again) as well as the two reduced systems[^3]:

[^3]: All of this is done with `ImplicitMidpoint` as integrator.

```@example toda_lattice 
integrate_full_system(psd_rs) # hide
integrate_reduced_system(psd_rs) # hide
integrate_reduced_system(sae_rs) # hide

@time "FOM + Implicit Midpoint" sol_full = integrate_full_system(psd_rs) # hide
@time "PSD + Implicit Midpoint" sol_psd_reduced = integrate_reduced_system(psd_rs) # hide
@time "SAE + Implicit Midpoint" sol_sae_reduced = integrate_reduced_system(sae_rs) # hide

nothing # hide
```

And plot the solutions for 

```@example toda_lattice
time_steps = (0, 300, 800)
nothing # hide
```

```@setup toda_lattice
using CairoMakie

morange = RGBf(255 / 256, 127 / 256, 14 / 256)
mred = RGBf(214 / 256, 39 / 256, 40 / 256) 
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)

# plot validation
function plot_validation!(fig, coordinates::Tuple, t_steps::Integer=100; theme = :dark)
    textcolor = theme == :dark ? :white : :black
    ax_val = Axis(fig[coordinates[1], coordinates[2]]; backgroundcolor = :transparent,
                                                                bottomspinecolor = textcolor, 
                                                                topspinecolor = textcolor,
                                                                leftspinecolor = textcolor,
                                                                rightspinecolor = textcolor,
                                                                xtickcolor = textcolor, 
                                                                ytickcolor = textcolor,
                                                                xticklabelcolor = textcolor,
                                                                yticklabelcolor = textcolor,
                                                                xlabel=L"\omega", 
                                                                ylabel=L"q",
                                                                xlabelcolor = textcolor,
                                                                ylabelcolor = textcolor)
    lines!(ax_val, Ω, sol_full.s.q[t_steps], label = rich("FOM + Implicit Midpoint"; color = textcolor), color = mblue)
    lines!(ax_val, Ω, psd_rs.decoder((q = sol_psd_reduced.s.q[t_steps], p = sol_psd_reduced.s.p[t_steps])).q, 
        label = rich("PSD + Implicit Midpoint"; color = textcolor), color = morange)
    lines!(ax_val, Ω, sae_rs.decoder((q = sol_sae_reduced.s.q[t_steps], p = sol_sae_reduced.s.p[t_steps])).q, 
        label = rich("SAE + Implicit Midpoint"; color = textcolor), color = mgreen)

    if t_steps == 0
        axislegend(ax_val; position = (1.01, 1.5), backgroundcolor = theme == :dark ? :transparent : :white, color = textcolor, labelsize = 8)
    end
    nothing
end

fig_light = Figure(; backgroundcolor = :transparent)

fig_dark = Figure(; backgroundcolor = :transparent)

for (i, time) in zip(1:length(time_steps), time_steps)
    plot_validation!(fig_light, (i, 1), time; theme = :light)
    plot_validation!(fig_dark, (i, 1), time; theme = :dark)
end

# axislegend(fig_light; position = (.82, .75), backgroundcolor = :transparent, color = :black)
# axislegend(fig_dark;  position = (.82, .75), backgroundcolor = :transparent, color = :white)

save("sae_validation.png", fig_light; px_per_unit = 1.2)
save("sae_validation_dark.png", fig_dark; px_per_unit = 1.2)

nothing # hide
```

```@example
Main.include_graphics("sae_validation"; width = .78, caption = raw"Comparison between FOM (blue), PSD with implicit midpoint (orange) and SAE with implicit midpoint (green). ") # hide
```

We can see that the SAE has much more approximation capabilities than the PSD. But even though the SAE reasonably reproduces the full-order model (FOM), we see that the online stage of the SAE takes even longer than evaluating the FOM. In order to solve this problem we have to make the *online stage more efficient*.

## The online stage with a neural network

Instead of using a standard integrator we can also use a neural network that is trained on the reduced data. For this: 

```@example toda_lattice
backend = CPU() # hide
const integrator_train_epochs = 65536
const integrator_batch_size = 4096
const seq_length = 4

integrator_architecture = StandardTransformerIntegrator(reduced_dim; 
                                                                    transformer_dim = 20, 
                                                                    n_blocks = 3, 
                                                                    n_heads = 5, 
                                                                    L = 3,
                                                                    upscaling_activation = tanh)

integrator_nn = NeuralNetwork(integrator_architecture, backend)

integrator_method = AdamOptimizerWithDecay(integrator_train_epochs)

o_integrator = Optimizer(integrator_method, integrator_nn)

dl = dl_cpu # hide
# map from autoencoder type to integrator type
dl_integration = DataLoader(dl; autoencoder = false)

integrator_batch = Batch(integrator_batch_size, seq_length)
nothing # hide
```
```julia
loss = GeometricMachineLearning.ReducedLoss(encoder(sae_nn_gpu), decoder(sae_nn_gpu))

train_integrator_loss = o_integrator(   integrator_nn, 
                                        dl_integration, 
                                        integrator_batch, 
                                        integrator_train_epochs, 
                                        loss)
```

We can now evaluate the solution:

```@example toda_lattice
nn_integrator_parameters = load("integrator_parameters.jld2")["integrator_parameters"] # hide
integrator_nn = NeuralNetwork(integrator_architecture, Chain(integrator_architecture), _nnp(nn_integrator_parameters), backend) # hide
ics = encoder(sae_nn_cpu)((q = dl.input.q[:, 1:seq_length, 1], p = dl.input.p[:, 1:seq_length, 1])) # hide
iterate(mtc(integrator_nn), ics; n_points = length(sol.t), prediction_window = seq_length) # hide
@time "time stepping with transformer" time_series = iterate(mtc(integrator_nn), ics; n_points = length(sol.t), prediction_window = seq_length)
nothing # hide
```

```@setup toda_lattice
# plot validation
function plot_transformer_validation!(fig, coordinates, t_steps::Integer=100; theme = :dark)
    textcolor = theme == :dark ? :white : :black
    ax_val = Axis(fig[coordinates[1], coordinates[2]]; backgroundcolor = :transparent,
                                                                bottomspinecolor = textcolor, 
                                                                topspinecolor = textcolor,
                                                                leftspinecolor = textcolor,
                                                                rightspinecolor = textcolor,
                                                                xtickcolor = textcolor, 
                                                                ytickcolor = textcolor,
                                                                xticklabelcolor = textcolor,
                                                                yticklabelcolor = textcolor,
                                                                xlabel = L"\omega", 
                                                                ylabel = L"q",
                                                                xlabelcolor = textcolor,
                                                                ylabelcolor = textcolor)
    lines!(ax_val, Ω, sol_full.s.q[t_steps], label = rich("FOM + Implicit Midpoint"; color = textcolor), color = mblue)
    lines!(ax_val, Ω, psd_rs.decoder((q = sol_psd_reduced.s.q[t_steps], p = sol_psd_reduced.s.p[t_steps])).q, 
        label = rich("PSD + Implicit Midpoint"; color = textcolor), color = morange)
    lines!(ax_val, Ω, sae_rs.decoder((q = sol_sae_reduced.s.q[t_steps], p = sol_sae_reduced.s.p[t_steps])).q, 
        label = rich("SAE + Implicit Midpoint"; color = textcolor), color = mgreen)

    time_series = iterate(mtc(integrator_nn), ics; n_points = t_steps, prediction_window = seq_length)
    # prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
    prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
    sol = decoder(sae_nn_cpu)(prediction)

    lines!(ax_val, Ω, sol.q; label = rich("SAE + Transformer"; color = textcolor), color = mpurple)

    if t_steps == 0
        axislegend(ax_val; position = (1.01, .9), backgroundcolor = theme == :dark ? :transparent : :white, color = textcolor, labelsize = 8, nbanks = 2)
    end
    nothing
end

fig_light = Figure(; backgroundcolor = :transparent)

fig_dark = Figure(; backgroundcolor = :transparent)

for (i, time) in zip(1:length(time_steps), time_steps)
    plot_transformer_validation!(fig_light, (i, 1), time; theme = :light)
    plot_transformer_validation!(fig_dark, (i, 1), time; theme = :dark)
end

save("sae_integrator_validation.png", fig_light; px_per_unit = 1.2)
save("sae_integrator_validation_dark.png", fig_dark; px_per_unit = 1.2)

nothing # hide
```

```@example
Main.include_graphics("sae_integrator_validation"; width = .78, caption = raw"Comparison between FOM (blue), PSD with implicit midpoint (orange), SAE with implicit midpoint (green) and SAE with transformer (purple). ") # hide
```

Note that integration of the system with the transformer is orders of magnitudes faster than any comparable method and also leads to an improvement in accuracy over the case where we build the reduced space with the symplectic autoencoder and use implicit midpoint in the online phase.

```@eval
Main.remark(raw"While training the symplectic autoencoder we completely ignore the online phase, but only aim at finding a good low-dimensional approximation to the solution manifold. This is why we observe that the approximated solution differs somewhat form the actual one when using implicit midpoint for integrating the low-dimensional system (blue line vs. green line).")
```

```@raw latex
\begin{comment}
```
We can also make an animation of the resulting solution using `Makie` [DanischKrumbiegel2021](@cite):

```@setup toda_lattice
time_steps = 0:10:(length(sol.q) * 10)

time_series = iterate(mtc(integrator_nn), ics; n_points = length(sol.t) * 10, prediction_window = seq_length)
time_steps = axes(time_series.q, 2)
function make_animation(; theme = :dark)
textcolor = theme == :dark ? :white : :black 
fig = Figure()
ax = Axis(fig[1, 1],    backgroundcolor = :transparent,
                        bottomspinecolor = textcolor, 
                        topspinecolor = textcolor,
                        leftspinecolor = textcolor,
                        rightspinecolor = textcolor,
                        xtickcolor = textcolor, 
                        ytickcolor = textcolor,
                        xticklabelcolor = textcolor,
                        yticklabelcolor = textcolor,
                        xlabel=L"\omega", 
                        ylabel=L"q",
                        xlabelcolor = textcolor,
                        ylabelcolor = textcolor)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
framerate = 30
mblue = 
record(fig, "toda_animation.mp4", time_steps;
    framerate = framerate) do time_step
    empty!(ax)
    time_step < length(sol.t) ? lines!(ax, sol.q[time_step, :], color = mblue) : nothing
    prediction = (q = time_series.q[:, time_step], p = time_series.p[:, time_step])
    sol_sae_t = decoder(sae_nn_cpu)(prediction)
    lines!(ax, sol_sae_t.q, color = mpurple, label = "t = $(sol.t[time_step])")
    ylims!(ax, 0., 1.)
    axislegend(ax; position = (1.01, 1.5), labelsize = 8)
end
nothing
end
make_animation(; theme = :dark)
make_animation(; theme = :light)
nothing # hide
```

```@example
Docs.HTML("""<video mute autoplay loop controls src="toda_animation.mp4" />""") # hide
```


Here we compared PSD with an SAE whith the same reduced dimension. One may argue that this is not entirely fair as the PSD has much fewer parameters than the SAE:

```@example toda_lattice
(parameterlength(psd_nn_cpu), parameterlength(sae_nn_cpu))
```

and we also saw that evaluating *PSD + Implicit Midpoint* is much faster than *SAE + Implicit Midpoint*. We thus model the system with PSDs of higher reduced dimension:

```@example toda_lattice
const reduced_dim2 = 8

Random.seed!(123) # hide
psd_arch2 = PSDArch(dl_cpu.input_dim, reduced_dim2)

psd_nn2 = NeuralNetwork(psd_arch2, CPU(), eltype(dl_cpu))

solve!(psd_nn2, dl_cpu)
```

And we see that the error is a lot lower than for the case `reduced_dim = 2`. We now proceed with building the reduced Hamiltonian system as before, again using [`HRedSys`](@ref):

```@example toda_lattice
psd_rs2 = HRedSys(pr, encoder(psd_nn2), decoder(psd_nn2); integrator = ImplicitMidpoint())
nothing # hide
```

We integrate this PSD to check how big the difference in performance is:
```@example toda_lattice
integrate_reduced_system(psd_rs2) # hide
@time "PSD + Implicit Midpoint" sol_psd_reduced2 = integrate_reduced_system(psd_rs2) # hide
nothing # hide
```

We can also plot the comparison with the FOM as before:

```@setup toda_lattice
morange = RGBf(255 / 256, 127 / 256, 14 / 256)
mred = RGBf(214 / 256, 39 / 256, 40 / 256) 
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)

# plot validation
function plot_validation!(fig, coordinates::Tuple, t_steps::Integer=100; theme = :dark)
    textcolor = theme == :dark ? :white : :black
    ax_val = Axis(fig[coordinates[1], coordinates[2]]; backgroundcolor = :transparent,
                                                                bottomspinecolor = textcolor, 
                                                                topspinecolor = textcolor,
                                                                leftspinecolor = textcolor,
                                                                rightspinecolor = textcolor,
                                                                xtickcolor = textcolor, 
                                                                ytickcolor = textcolor,
                                                                xticklabelcolor = textcolor,
                                                                yticklabelcolor = textcolor,
                                                                xlabel=L"\omega", 
                                                                ylabel=L"q",
                                                                xlabelcolor = textcolor,
                                                                ylabelcolor = textcolor)
    lines!(ax_val, Ω, sol_full.s.q[t_steps], label = rich("FOM + Implicit Midpoint"; color = textcolor), color = mblue)
    lines!(ax_val, Ω, psd_rs2.decoder((q = sol_psd_reduced2.s.q[t_steps], p = sol_psd_reduced2.s.p[t_steps])).q, 
        label = rich("PSD + Implicit Midpoint"; color = textcolor), color = morange)

    if t_steps == 0
        axislegend(ax_val; position = (1.01, 1.5), backgroundcolor = theme == :dark ? :transparent : :white, color = textcolor, labelsize = 8)
    end
    nothing
end

fig_light = Figure(; backgroundcolor = :transparent)

fig_dark = Figure(; backgroundcolor = :transparent)

for (i, time) in zip(1:length(time_steps), time_steps)
    plot_validation!(fig_light, (i, 1), time; theme = :light)
    plot_validation!(fig_dark, (i, 1), time; theme = :dark)
end

# axislegend(fig_light; position = (.82, .75), backgroundcolor = :transparent, color = :black)
# axislegend(fig_dark;  position = (.82, .75), backgroundcolor = :transparent, color = :white)

save("psd_validation2.png", fig_light; px_per_unit = 1.2)
save("psd_validation2_dark.png", fig_dark; px_per_unit = 1.2)

nothing # hide
```

```@example
Main.include_graphics("psd_validation2"; width = .78, caption = raw"Comparison between the FOM and the PSD with a bigger reduced dimension. ") # hide
```

We see that for a reduced dimension of ``2n = 8`` the PSD looks slightly better than the SAE for ``2n = 2.`` As with the SAE we can also use a transformer to integrate the dynamics on the low-dimensional space:

```@example toda_lattice
const integrator_architecture2 = StandardTransformerIntegrator(reduced_dim2; 
                                                                            transformer_dim = 20, 
                                                                            n_blocks = 3, 
                                                                            n_heads = 5, 
                                                                            L = 3, 
                                                                            upscaling_activation = tanh)
integrator_nn2 = NeuralNetwork(integrator_architecture2, backend)
const integrator_method2 = AdamOptimizerWithDecay(integrator_train_epochs)
const o_integrator2 = Optimizer(integrator_method2, integrator_nn2)

loss2 = GeometricMachineLearning.ReducedLoss(encoder(psd_nn2), decoder(psd_nn2))
nothing # hide
```

For training we leave `dl_integration`, `integrator_batch` and `integrator_train_epochs` unchanged:
```julia
train_integrator_loss2 = o_integrator(integrator_nn2, dl_integration, integrator_batch, integrator_train_epochs, loss2)
```

```@setup toda_lattice
nn_integrator_parameters2 = load("integrator_parameters_psd.jld2")["integrator_parameters"] # hide
integrator_nn2 = NeuralNetwork(integrator_architecture2, Chain(integrator_architecture2), _nnp(nn_integrator_parameters2), backend) # hide
ics = encoder(psd_nn2)((q = dl_cpu.input.q[:, 1:seq_length, 1], p = dl_cpu.input.p[:, 1:seq_length, 1])) # hide
nothing # hide
```

We again integrate the system and then plot the result:

```@example toda_lattice
iterate(mtc(integrator_nn2), ics; n_points = length(sol.t), prediction_window = seq_length) # hide
@time "time stepping with transformer" time_series2 = iterate(mtc(integrator_nn2), ics; n_points = length(sol.t), prediction_window = seq_length)
nothing # hide
```

We see that using the transformer on the six-dimensional PSD-reduced system takes slightly longer than using the transformer on the two-dimensional SAE-reduced system. The accuracy is much worse however. Before we plotted the solution for:

```@example toda_lattice
time_steps
```

Now we do so with:

```@example toda_lattice
time_steps = (0, 4, 5)
nothing # hide
```

```@setup toda_lattice
# plot validation
function plot_validation!(fig, coordinates::Tuple, t_steps::Integer=100; theme = :dark)
    textcolor = theme == :dark ? :white : :black
    ax_val = Axis(fig[coordinates[1], coordinates[2]]; backgroundcolor = :transparent,
                                                                bottomspinecolor = textcolor, 
                                                                topspinecolor = textcolor,
                                                                leftspinecolor = textcolor,
                                                                rightspinecolor = textcolor,
                                                                xtickcolor = textcolor, 
                                                                ytickcolor = textcolor,
                                                                xticklabelcolor = textcolor,
                                                                yticklabelcolor = textcolor,
                                                                xlabel=L"\omega", 
                                                                ylabel=L"q",
                                                                xlabelcolor = textcolor,
                                                                ylabelcolor = textcolor)
    lines!(ax_val, Ω, sol_full.s.q[t_steps], label = rich("FOM + Implicit Midpoint"; color = textcolor), color = mblue)
    lines!(ax_val, Ω, psd_rs2.decoder((q = sol_psd_reduced2.s.q[t_steps], p = sol_psd_reduced2.s.p[t_steps])).q, 
        label = rich("PSD + Implicit Midpoint"; color = textcolor), color = morange)

    time_series2 = iterate(mtc(integrator_nn2), ics; n_points = t_steps, prediction_window = seq_length)
    # prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
    prediction2 = (q = time_series2.q[:, end], p = time_series2.p[:, end])
    sol = decoder(psd_nn2)(prediction2)

    lines!(ax_val, Ω, sol.q; label = rich("PSD + Transformer"; color = textcolor), color = mred)

    if t_steps == 0
        axislegend(ax_val; position = (1.01, 1.5), backgroundcolor = theme == :dark ? :transparent : :white, color = textcolor, labelsize = 8)
    end
    nothing
end

fig_light = Figure(; backgroundcolor = :transparent)

fig_dark = Figure(; backgroundcolor = :transparent)

for (i, time) in zip(1:length(time_steps), time_steps)
    plot_validation!(fig_light, (i, 1), time; theme = :light)
    plot_validation!(fig_dark, (i, 1), time; theme = :dark)
end

# axislegend(fig_light; position = (.82, .75), backgroundcolor = :transparent, color = :black)
# axislegend(fig_dark;  position = (.82, .75), backgroundcolor = :transparent, color = :white)

save("psd_integrator_validation.png", fig_light; px_per_unit = 1.2)
save("psd_integrator_validation_dark.png", fig_dark; px_per_unit = 1.2)

nothing # hide
```

```@example
Main.include_graphics("psd_integrator_validation"; width = .78, caption = raw"Comparison between FOM (blue), PSD with implicit midpoint (orange), and PSD with transformer (red). ") # hide
```

Here we however see a dramatic deterioration in the quality of the approximation. We assume that this because the `transformer_dim` was chosen to be `20` for the SAE and the PSD, but in the second case the reduced space is of dimension six, whereas it is of dimension two in the first case. This may mean that we need an even bigger transformer to find a good approximation of the reduced space.

```@raw latex
\end{comment}
```


```@raw latex
\section*{Chapter Summary}
We showed that for the Toda lattice we can achieve a very good approximation to the full-order system by using a two-dimensional reduced space; we also showed that for such a small space proper symplectic decomposition utterly fails.

We however also saw that when we use standard integrators in the online phase, the very low dimension of the reduced space may not mean fast computation. But by using a transformer as a neural network-based integrator we could circumvent this problem and achieve a speed-up of a factor of approximately 1000.

\begin{comment}
```

## References 

```@bibliography
Pages = []
Canonical = false

peng2016symplectic
greif2019decay
buchfink2023symplectic
brantner2023symplectic
```

```@raw latex
\end{comment}
```

```@raw html
<!--
```

# References 
```@bibliography
Pages = []
Canonical = false

peng2016symplectic
greif2019decay
buchfink2023symplectic
brantner2023symplectic
```


```@raw html
-->
```