# Symplectic Autoencoders and the Toda Lattice

In this tutorial we use a [SymplecticAutoencoder](@ref) to approximate the linear wave equation with a lower-dimensional Hamiltonian model and compare it with standard proper symplectic decomposition (PSD).

## The system

The [Toda lattice](https://juliagni.github.io/GeometricProblems.jl/latest/toda_lattice) is a prototypical example of a Hamiltonian PDE. It is described by 
```math
    H(q, p) = \sum_{n\in\mathbb{Z}}\left(  \frac{p_n^2}{2} + \alpha e^{q_n - q_{n+1}} \right).
```

We further assume a finite number of particles ``N`` and impose periodic boundary conditions: 
```math
\begin{aligned}
    q_{n+N} &  \equiv q_n \\ 
    p_{n+N} &   \equiv p_n.
\end{aligned}
```

In this tutorial we want to reduce the dimension of the big system by a significant factor with (i) proper symplectic decomposition (PSD) and (ii) symplectic autoencoders. The first approach is strictly linear whereas the second one allows for more general mappings. 

### Using the Toda lattice in numerical experiments 

In order to use the Toda lattice in numerical experiments we have to pick suitable initial conditions. For this, consider the [third-degree spline](https://juliagni.github.io/GeometricProblems.jl/latest/initial_condition): 

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
u_0(\mu)(\omega) = h(s(\omega, \mu)), \quad s(\omega, \mu) =  20 \mu  |\omega + \frac{\mu}{2}|.
```

For the purposes of this tutorial we will use the default value for ``\mu`` provided in `GeometricMachineLearning`:

```@example
using GeometricProblems.TodaLattice: μ

μ
```

## Get the data 

The training data can very easily be obtained by using the packages [`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl) and [`GeometricIntegrators`](https://github.com/JuliaGNI/GeometricIntegrators.jl):

```@example toda_lattice
using GeometricProblems.TodaLattice: hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricMachineLearning 
using Plots
import Random

pr = hodeproblem(; tspan = (0.0, 100.))
sol = integrate(pr, ImplicitMidpoint())
dl = DataLoader(sol; autoencoder = true)

dl.input_dim
```

Here we first integrate the system with implicit midpoint and then put the training data into the right format by calling `DataLoader`. We can get the dimension of the system by calling `dl.input_dim`. Also note that the keyword `autoencoder` was set to true.

## Train the network 

We now want to compare two different approaches: [PSDArch](@ref) and [SymplecticAutoencoder](@ref). For this we first have to set up the networks: 

```@example toda_lattice
const reduced_dim = 2

psd_arch = PSDArch(dl.input_dim, reduced_dim)
sae_arch = SymplecticAutoencoder(dl.input_dim, reduced_dim; n_encoder_blocks = 4, n_decoder_blocks = 4, n_encoder_layers = 4, n_decoder_layers = 1)

Random.seed!(123)
psd_nn = NeuralNetwork(psd_arch)
sae_nn = NeuralNetwork(sae_arch)

nothing # hide
```

Training a neural network is usually done by calling an instance of [Optimizer](@ref) in `GeometricMachineLearning`. [PSDArch](@ref) however can be solved directly by using singular value decomposition and this is done by calling [solve!](@ref). The `SymplecticAutoencoder` we train with the [AdamOptimizer](@ref) however: 

```@example toda_lattice 
const n_epochs = 8
const batch_size = 16

o = Optimizer(sae_nn, AdamOptimizer(Float64))

psd_error = solve!(psd_nn, dl)
sae_error = o(sae_nn, dl, Batch(batch_size), n_epochs)

hline([psd_error]; color = 2, label = "PSD error")
plot!(sae_error; color = 3, label = "SAE error", xlabel = "epoch", ylabel = "training error")
```

## The online stage with a standard integrator

After having trained our neural network we can now evaluate it in the online stage of reduced complexity modeling: 

```@example toda_lattice
psd_rs = HRedSys(pr, encoder(psd_nn), decoder(psd_nn); integrator = ImplicitMidpoint())
sae_rs = HRedSys(pr, encoder(sae_nn), decoder(sae_nn); integrator = ImplicitMidpoint())

projection_error(psd_rs)
```

```@example toda_lattice 
projection_error(sae_rs)
```

Next we plot a comparison between the PSD prediction and the symplectic autoencoder prediction: 

```@example toda_lattice
sol_full = integrate_full_system(psd_rs)
sol_psd_reduced = integrate_reduced_system(psd_rs)
sol_sae_reduced = integrate_reduced_system(sae_rs)

const t_steps = 100
plot(sol_full.s.q[t_steps], label = "Implicit Midpoint")
plot!(psd_rs.decoder((q = sol_psd_reduced.s.q[t_steps], p = sol_psd_reduced.s.p[t_steps])).q, label = "PSD")
plot!(sae_rs.decoder((q = sol_sae_reduced.s.q[t_steps], p = sol_sae_reduced.s.p[t_steps])).q, label = "SAE")
```

We can see that the autoencoder approach has much more approximation capabilities than the psd approach. The jiggly lines are due to the fact that training was done for only 8 epochs. 

## The online stage with a neural network

Instead of using a standard integrator we can also use a neural network that is trained on the reduced data. For this: 

```@example toda_lattice
integrator_batch_size = 128
integrator_train_epochs = 10

integrator_nn = NeuralNetwork(GSympNet(reduced_dim))
o_integrator = Optimizer(AdamOptimizer(Float64), integrator_nn)

loss = ReducedLoss(encoder(sae_nn), decoder(sae_nn))
dl_integration = DataLoader(dl; autoencoder = false)

o_integrator(integrator_nn, dl_integration, Batch(integrator_batch_size), integrator_train_epochs, loss)

nothing # hide
```

We can now evaluate the solution:

```@example toda_lattice
ics = encoder(sae_nn)((q = dl.input.q[:, 1, 1], p = dl.input.p[:, 1, 1]))
time_series = iterate(integrator_nn, ics; n_points = t_steps)
prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
sol = decoder(sae_nn)(prediction)

plot!(sol.q; label = "Neural Network Integrator")
```

```@eval
Main.remark(raw"The results presented here are not ideal. In order to be able to quickly run them on a relatively weak gpu we have chosen the number of epochs very low. For more useful results the number of epochs should be increased to at least 1000 or more.")
```


## References 
```@bibliography
Pages = []
Canonical = false

buchfink2023symplectic
peng2016symplectic
greif2019decay
```