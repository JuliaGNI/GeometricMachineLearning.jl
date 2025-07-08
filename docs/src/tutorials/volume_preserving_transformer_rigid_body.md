# The Volume-Preserving Transformer for the Rigid Body

Here we train a [volume-preserving feedforward neural network](@ref "Volume-Preserving Feedforward Neural Network"), a [standard transformer](@ref "Standard Transformer") and a [volume-preserving transformer](@ref "Volume-Preserving Transformer") on a rigid body [hairer2006geometric, arnold1978mathematical](@cite). These are also the results presented in [brantner2024volume](@cite). The ODE that describes the rigid body is: 

```math
\frac{d}{dt}\begin{pmatrix} z_1 \\ z_2 \\ z_3 \end{pmatrix} = \begin{pmatrix} Az_2z_3 \\ Bz_1z_3 \\ Cz_1z_2 \end{pmatrix}.
```

In the following we use ``A = 1,`` ``B = 1/2`` and ``C = -1/2.`` For a derivation of this equation see [brantner2024volume](@cite). 

We first generate the data. The initial conditions that we use are:

```math
\mathtt{ics} = \left\{ \begin{pmatrix} \sin(\alpha) \\ 0 \\ \cos(\alpha) \end{pmatrix}, \begin{pmatrix} 0 \\ \sin(\alpha) \\ \cos(\alpha) \end{pmatrix}: \alpha = 0.1\mathtt{:}0.01\mathtt{:}2\pi \right\}.
```

We build these initial conditions by concatenating ``\mathtt{ics}_1`` and ``\mathtt{ics}_2``:

```@example rigid_body
const ics₁ = [[sin(val), 0., cos(val)] for val in .1:.01:(2*π)]
const ics₂ = [[0., sin(val), cos(val)] for val in .1:.01:(2*π)]
const ics = [ics₁..., ics₂...]
nothing # hide
```

We now generate the data by integrating with:

```@example rigid_body
const timestep = .2
const timespan = (0., 20.)
nothing # hide
```

The rigid body is implemented in [`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl):

```@example rigid_body
using GeometricMachineLearning # hide
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.RigidBody: odeproblem, odeensemble, default_parameters

ensemble_problem = odeensemble(ics; timespan = timespan, timestep = timestep, parameters = default_parameters)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl_cpu = DataLoader(ensemble_solution; suppress_info = true)
nothing # hide
```

We plot the trajectories for some of the initial conditions to get and idea of what the data look like:

```@example rigid_body
import Random # hide
Random.seed!(123456) # hide
const n_trajectories_to_plot = 5
indices = Int.(ceil.(size(dl_cpu.input, 3) * rand(n_trajectories_to_plot)))

trajectories = [dl_cpu.input[:, :, index] for index in indices]
nothing # hide
```

```@setup rigid_body
using GLMakie
include("../../gl_makie_transparent_background_hack.jl")
GLMakie.activate!() # hide

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256) # hide
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
colors = (morange, mred, mpurple, mblue, mgreen)
function set_up_plot(; theme = :dark) # hide
text_color = theme == :dark ? :white : :black # hide
fig = Figure(; backgroundcolor = :transparent, size = (900, 675)) # hide
ax = Axis3(fig[1, 1]; # hide
    backgroundcolor = (:tomato, .5), # hide
    aspect = (1., 1., 1.), # hide
    xlabel = L"z_1", # hide
    ylabel = L"z_2", # hide
    zlabel = L"z_3", # hide
    xgridcolor = text_color, # hide
    ygridcolor = text_color, # hide
    zgridcolor = text_color, # hide
    xtickcolor = text_color, # hide
    ytickcolor = text_color, # hide
    ztickcolor = text_color, # hide
    xlabelcolor = text_color, # hide
    ylabelcolor = text_color, # hide
    zlabelcolor = text_color, # hide
    xypanelcolor = :transparent, # hide
    xzpanelcolor = :transparent, # hide
    yzpanelcolor = :transparent, # hide
    limits = ([-1, 1], [-1, 1], [-1, 1]), # hide
    azimuth = π / 7, # hide
    elevation = π / 7, # hide
    # height = 75., # hide
    ) # hide
# plot a sphere with radius one and origin 0
surface!(ax, Main.sphere(1., [0., 0., 0.])...; alpha = .45, transparency = true)

for (trajectory, color, i) in zip(trajectories, colors, 1:length(trajectories))
    lines!(ax, trajectory[1, :], trajectory[2, :], trajectory[3, :]; color = color, linewidth = 4, label = "Trajectory $(i)")
end

axislegend(; position = (.82, .75), backgroundcolor = theme == :dark ? :transparent : :white, labelcolor = text_color) # hide
fig, ax # hide
end # hide

fig_light = set_up_plot(; theme = :light)[1] # hide
fig_dark = set_up_plot(; theme = :dark)[1] # hide

save("rigid_body_trajectories_light.png", alpha_colorbuffer(fig_light)) # hide
save("rigid_body_trajectories_dark.png", alpha_colorbuffer(fig_dark)) # hide

nothing # hide
```

![A sample of rigid body trajectories. This system has two conserved quantities.](rigid_body_trajectories_light.png)
![A sample of rigid body trajectories. This system has two conserved quantities.](rigid_body_trajectories_dark.png)

The rigid body has two conserved quantities:
1. one conserved quantity is the [Hamiltonian of the system](@ref "Symplectic Systems"): ``H(z_1, z_2, z_3) = \frac{1}{2}\left( \frac{z_1^2}{I_1} + \frac{z_2^2}{I_2} + \frac{z_3^2}{I_3} \right),``
2. the second one is the quadratic invariant: ``I(z_1, z_2, z_3) = z_1^2 + z_2^2 + z_3^2.``

The coefficients ``I_1,`` ``I_2`` and ``I_3`` can be obtained through

```math
\begin{aligned}
A = \frac{I_2 - I_3}{I_2I_3}, \\ 
B = \frac{I_3 - I_1}{I_3I_1}, \\ 
C = \frac{I_1 - I_2}{I_1I_2}.
\end{aligned}
```

The second conserved invariant ``I(\cdot, \cdot, \cdot)`` is visualized through the sphere in the figure above. The conserved Hamiltonian is the reason for why the curves are closed.

The rigid body has Poisson structure [hairer2006geometric](@cite), but does not have canonical Hamiltonian structure. We can thus not use [SympNets](@ref "SympNet Architecture") or [symplectic transformers](@ref "Linear Symplectic Transformer") here, but the ODE is clearly divergence-free. We use this to demonstrate the efficacy of the [volume-preserving transformer](@ref "Volume-Preserving Transformer"). We set up our networks:

```@example rigid_body
# hyperparameters concerning the architectures 
const sys_dim = size(dl_cpu.input, 1)
const n_heads = 1
const L = 3 # transformer blocks 
const activation = tanh
const resnet_activation = tanh
const n_linear = 1
const n_blocks = 2
const skew_sym = false
const seq_length = 3

arch_vpff = VolumePreservingFeedForward(sys_dim, n_blocks * L, n_linear, resnet_activation)
arch_vpt = VolumePreservingTransformer(sys_dim, seq_length; 
                                            n_blocks = n_blocks, 
                                            n_linear = n_linear, 
                                            L = L, 
                                            activation = resnet_activation, 
                                            skew_sym = skew_sym)
arch_st = StandardTransformerIntegrator(sys_dim;    n_heads = n_heads, 
                                                    transformer_dim = sys_dim, 
                                                    n_blocks = n_blocks, 
                                                    L = L, 
                                                    resnet_activation = resnet_activation, 
                                                    add_connection = false)
nothing # hide
```

Note that we set the keyword `skew_sym` to `false` here. This is different from what we did in [brantner2024volume](@cite), where it was set to true[^1]. We allocate the networks on GPU:

[^1]: A detailed discussion of the consequences of setting this keyword is presented [as a separate example](@ref "Comparing Different `VolumePreservingAttention` Mechanisms").

```julia
using CUDA
backend = CUDABackend()
```
```@example rigid_body
T = Float32
backend = CPU() # hide

dl = DataLoader(dl_cpu, backend, T)
nn_vpff = NeuralNetwork(arch_vpff, backend, T)
nn_vpt = NeuralNetwork(arch_vpt, backend, T)
nn_st = NeuralNetwork(arch_st, backend, T)

(parameterlength(nn_vpff), parameterlength(nn_vpt), parameterlength(nn_st))
```

We now train the various networks. For this we use [`AdamOptimizerWithDecay`](@ref):

```@example rigid_body
const n_epochs = 500000
const batch_size = 16384
const feedforward_batch = Batch(batch_size)
const transformer_batch = Batch(batch_size, seq_length, seq_length)
const opt_method = AdamOptimizerWithDecay(n_epochs, T; η₁ = 1e-2, η₂ = 1e-6)

o_vpff = Optimizer(opt_method, nn_vpff)
o_vpt = Optimizer(opt_method, nn_vpt)
o_st = Optimizer(opt_method, nn_st)
nothing # hide
```
```julia
o_vpff(nn_vpff, dl, feedforward_batch, n_epochs)
o_vpt(nn_vpt, dl, transformer_batch, n_epochs)
o_st(nn_st, dl, transformer_batch, n_epochs)
```

After the networks have been trained we map the parameters to cpu:

```@example rigid_body
const mtc = GeometricMachineLearning.map_to_cpu
nn_vpff = mtc(nn_vpff)
nn_vpt = mtc(nn_vpt)
nn_st = mtc(nn_st)
using JLD2 # hide
# get correct parameters from jld2 file # hide
f = load("transformer_rigid_body.jld2")  # hide
_nnp(ps::Tuple) = NeuralNetworkParameters{Tuple(Symbol("L$(i)") for i in 1:length(ps))}(ps) # hide
nn_vpff = NeuralNetwork(nn_vpff.architecture, nn_vpff.model, _nnp(f["nn_vpff_params"]), nn_vpff.backend) # hide
nn_vpt = NeuralNetwork(nn_vpt.architecture, nn_vpt.model, _nnp(f["nn_vpt_arb_params"]), nn_vpt.backend) # hide
nn_st = NeuralNetwork(nn_st.architecture, nn_st.model, _nnp(f["nn_st_params"]), nn_st.backend) # hide
nothing # hide
```

After this we use [`iterate`](@ref) to obtain predicted orbits:

```@example rigid_body
ics_val₁ = [0., sin(0.9), cos(0.9 + π)]
ics_val₂ = [0., sin(1.1), cos(1.1)]
const t_validation = 120

function produce_trajectory(ics_val)
    problem = odeproblem(ics_val;   timespan = (0, t_validation), 
                                    timestep = timestep, 
                                    parameters = default_parameters)
    solution = integrate(problem, ImplicitMidpoint())
    trajectory = Float32.(DataLoader(solution; suppress_info = true).input)
    nn_vpff_solution = iterate(nn_vpff, trajectory[:, 1]; 
                                            n_points = Int(floor(t_validation / timestep)) + 1)
    nn_vpt_solution = iterate(nn_vpt, trajectory[:, 1:seq_length];
                                            n_points = Int(floor(t_validation / timestep)) + 1)
    nn_st_solution = iterate(nn_st, trajectory[:, 1:seq_length];
                                            n_points = Int(floor(t_validation / timestep)) + 1)
    trajectory, nn_vpff_solution, nn_vpt_solution, nn_st_solution
end

trajectory₁, nn_vpff_solution₁, nn_vpt_solution₁, nn_st_solution₁ = produce_trajectory(ics_val₁)
trajectory₂, nn_vpff_solution₂, nn_vpt_solution₂, nn_st_solution₂ = produce_trajectory(ics_val₂)
nothing # hide
```

```@setup rigid_body
function evaluate_plot!(fig, coordinate1, coordinate2, trajectory1, trajectory2; theme = :dark, label = "", color = mgreen) # hide
text_color = theme == :dark ? :white : :black # hide
ax = Axis3(fig[coordinate1, coordinate2]; # hide
    backgroundcolor = (:tomato, .5), # hide
    aspect = (1., 1., 1.), # hide
    xlabel = L"z_1", # hide
    ylabel = L"z_2", # hide
    zlabel = L"z_3", # hide
    xgridcolor = text_color, # hide
    ygridcolor = text_color, # hide
    zgridcolor = text_color, # hide
    xtickcolor = text_color, # hide
    ytickcolor = text_color, # hide
    ztickcolor = text_color, # hide
    xlabelcolor = text_color, # hide
    ylabelcolor = text_color, # hide
    zlabelcolor = text_color, # hide
    xypanelcolor = :transparent, # hide
    xzpanelcolor = :transparent, # hide
    yzpanelcolor = :transparent, # hide
    limits = ([-1, 1], [-1, 1], [-1, 1]), # hide
    azimuth = π / 7, # hide
    elevation = π / 7, # hide
    # this has to be fixed manually for now: https://discourse.julialang.org/t/makie-unwanted-whitespace-to-right-of-3d-subplots-and-dont-fully-understand-behaviour-of-colgap-with-3d-plots/108636
    protrusions = (50, 0, 0, 12)
    # height = 75., # hide
    ) # hide
# plot a sphere with radius one and origin 0
surface!(ax, Main.sphere(1., [0., 0., 0.])...; alpha = .45, transparency = true)

lines!(ax, trajectory₁[1, :], trajectory₁[2, :], trajectory₁[3, :], color = mblue, label = "Implicit Midpoint", linewidth = 2)
lines!(ax, trajectory₂[1, :], trajectory₂[2, :], trajectory₂[3, :], color = mblue, linewidth = 2)
lines!(ax, trajectory1[1, :], trajectory1[2, :], trajectory1[3, :], color = color, label = label, linewidth = 2)
lines!(ax, trajectory2[1, :], trajectory2[2, :], trajectory2[3, :], color = color, linewidth = 2)
axislegend(ax; position = (.82, 1.15), backgroundcolor = theme == :dark ? :transparent : :white, labelcolor = text_color) # hide
ax # hide
end # hide

fig_dark = Figure(; backgroundcolor = :transparent, size = (900, 675)) # hide
fig_light = Figure(; backgroundcolor = :transparent, size = (900, 675)) # hide

ax_dark_vpff = evaluate_plot!(fig_dark, 1, 1, nn_vpff_solution₁, nn_vpff_solution₂; theme = :dark, label = "VPFF", color = mgreen)
ax_dark_vpff = evaluate_plot!(fig_dark, 1, 2, nn_vpt_solution₁, nn_vpt_solution₂; theme = :dark, label = "VPT", color = mpurple)
ax_dark_vpff = evaluate_plot!(fig_dark, 1, 3, nn_st_solution₁, nn_st_solution₂; theme = :dark, label = "ST", color = morange)

ax_light_vpff = evaluate_plot!(fig_light, 1, 1, nn_vpff_solution₁, nn_vpff_solution₂; theme = :light, label = "VPFF", color = mgreen)
ax_light_vpff = evaluate_plot!(fig_light, 1, 2, nn_vpt_solution₁, nn_vpt_solution₂; theme = :light, label = "VPT", color = mpurple)
ax_light_vpff = evaluate_plot!(fig_light, 1, 3, nn_st_solution₁, nn_st_solution₂; theme = :light, label = "ST", color = morange)

rowsize!(fig_light.layout, 1, Aspect(1, 1.))
rowsize!(fig_dark.layout, 1, Aspect(1, 1.))
resize_to_layout!(fig_light)
resize_to_layout!(fig_dark)
save("rigid_body_evaluation_light.png", alpha_colorbuffer(fig_light))
save("rigid_body_evaluation_dark.png", alpha_colorbuffer(fig_dark))
nothing
```

![Evaluation of the three different networks for an interval (0, 120).](rigid_body_evaluation_light.png)
![Evaluation of the three different networks for an interval (0, 120).](rigid_body_evaluation_dark.png)

We can see that the volume-preserving transformer performs much better than the volume-preserving feedforward neural network and the standard transformer. It is especially noteworthy that its curves stick to the sphere at all times, which is not the case for the standard transformer. We also see that the standard transformer seems to perform better for one of the curves shown, but completely fails for the other. Why this is should be further investigated. 

We also compare the times it takes to integrate the system with (i) implicit midpoint, (ii) the volume-preserving transformer and (iii) the standard transformer:
```@example rigid_body
function timing() # hide
problem = odeproblem(ics_val₁; timespan = (0, t_validation), timestep = timestep, parameters = default_parameters) # hide
solution = integrate(problem, ImplicitMidpoint()) # hide
@time "Implicit Midpoint" solution = integrate(problem, ImplicitMidpoint())
trajectory = Float32.(DataLoader(solution; suppress_info = true).input)
iterate(nn_vpt, trajectory[:, 1:seq_length]; n_points = Int(floor(t_validation / timestep)) + 1) # hide
@time "VPT" iterate(nn_vpt, trajectory[:, 1:seq_length]; n_points = Int(floor(t_validation / timestep)) + 1)
iterate(nn_st, trajectory[:, 1:seq_length]; n_points = Int(floor(t_validation / timestep)) + 1) # hide
@time "ST" iterate(nn_st, trajectory[:, 1:seq_length]; n_points = Int(floor(t_validation / timestep)) + 1)
nothing # hide
end # hide
timing() # hide
```

Here we see that standard transformer is the fastest, followed by the volume-preserving transformers. Both neural network integrators are however faster than implicit midpoint (as evaluating them is completely explicit). The ODE we treated here is a very simple one. For [more complicated cases](@ref "Symplectic Autoencoders and the Toda Lattice") the speed-up we gain by using neural networks can be up to a factor 1000.