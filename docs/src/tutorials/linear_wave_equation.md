# Symplectic Autoencoders for the Linear Wave Equation

In this tutorial we use symplectic autoencoders to approximate the linear wave equation with a lower-dimensional Hamiltonian model and compare it with standard proper symplectic decomposition (PSD).

## Problem statement

The [linear wave equation](https://juliagni.github.io/GeometricProblems.jl/latest/linear_wave) is a prototypical example of a Hamiltonian PDE. It is given by (see [buchfink2023symplectic, peng2016symplectic](@cite)): 
```math
\mathcal{H}(q, p; \mu) := \frac{1}{2}\int_\Omega\mu^2(\partial_\xi{}q(t,\xi;\mu))^2 + p(t,\xi;\mu)^2d\xi,
```
with ``\xi\in\Omega:=(-1/2,1/2)`` and ``\mu\in\mathbb{P}:=[5/12,5/6]`` as a possible choice for domain and parameters. 

The PDE for to this Hamiltonian can be obtained similarly as in the ODE case:

```math
\partial_t{}q(t,\xi;\mu) = \frac{\delta{}\mathcal{H}}{\delta{}p} = p(t,\xi;\mu), \quad \partial_t{}p(t,\xi;\mu) = -\frac{\delta{}\mathcal{H}}{\delta{}q} = \mu^2\partial_{\xi{}\xi}q(t,\xi;\mu).
```

As with any other PDE, the wave equation can also be discretized to obtain a ODE which can be solved numerically.

If we discretize ``\mathcal{H}`` directly, to obtain a Hamiltonian on a finite-dimensional vector space ``\mathbb{R}^{2N}``, we get a Hamiltonian ODE[^1]:

[^1]: This conserves the Hamiltonian structure of the system.

```math
\mathcal{H}_h(z) = \sum_{i=1}^{\tilde{N}}\frac{\Delta{}x}{2}\bigg[p_i^2 + \mu^2\frac{(q_i - q_{i-1})^2 + (q_{i+1} - q_i)^2}{2\Delta{}x^2}\bigg]. 
```

The vector field of the FOM is described by [peng2016symplectic](@cite):

```math
  \frac{dz}{dt} = \mathbb{J}_d\nabla_z\mathcal{H}_h, \quad \mathbb{J}_d = \frac{\mathbb{J}_{2N}}{\Delta{}x}.
```

The wave equation has a slowely-decaying [Kolmogorov ``n``-width](../reduced_order_modeling/kolmogorov_n_width.md) [greif2019decay](@cite), which means linear methods like PSD will perform poorly.

### Using the Linear Wave Equation in Numerical Experiments 

In order to use the linear wave equation in numerical experiments we have to pick suitable initial conditions. For this, consider the [third-degree spline](https://juliagni.github.io/GeometricProblems.jl/latest/initial_condition): 

```math
h(s)  = \begin{cases}
        1 - \frac{3}{2}s^2 + \frac{3}{4}s^3 & \text{if } 0 \leq s \leq 1 \\ 
        \frac{1}{4}(2 - s)^3 & \text{if } 1 < s \leq 2 \\ 
        0 & \text{else.} 
\end{cases}
```

Plotted on the relevant domain it looks like this: 

```@example
import Images, Plots # hide
if Main.output_type == :html # hide
  HTML("""<object type="image/svg+xml" class="display-light-only" data=$(joinpath(Main.buildpath, "../tikz/third_degree_spline.png"))></object>""") # hide
else # hide
  Plots.plot(Images.load("../tikz/third_degree_spline.png"), axis=([], false)) # hide
end # hide
```

```@example
if Main.output_type == :html # hide
  HTML("""<object type="image/svg+xml" class="display-dark-only" data=$(joinpath(Main.buildpath, "../tikz/third_degree_spline_dark.png"))></object>""") # hide
end # hide
```


Taking the above function ``h(s)`` as a starting point, the initial conditions for the linear wave equations will now be constructed under the following considerations: 
- the initial condition (i.e. the shape of the wave) should depend on the parameter of the vector field, i.e. ``u_0(\mu)(\omega) = h(s(\omega, \mu))``.
- the solutions of the linear wave equation will travel with speed ``\mu``, and we should make sure that the wave does not *touch* the right boundary of the domain, i.e. 0.5. So the peak should be sharper for higher values of ``\mu`` as the wave will travel faster.
- the wave should start at the left boundary of the domain, i.e. at point 0.5, so to cover it as much as possible. 

Based on this we end up with the following choice of parametrized initial conditions: 

```math 
u_0(\mu)(\omega) = h(s(\omega, \mu)), \quad s(\omega, \mu) =  20 \mu  |\omega + \frac{\mu}{2}|.
```

## Get the data 

The training data can very easily be obtained by using the packages [`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl) and [`GeometricIntegrators`](https://github.com/JuliaGNI/GeometricIntegrators.jl):

```@example linear_wave
using GeometricProblems.LinearWave: hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricMachineLearning 

sol = integrate(hodeproblem(), ImplicitMidpoint())
dl = DataLoader(sol)

dl.input_dim
```

Here we first integrate the system with implicit midpoint and then put the training data into the right format by calling `DataLoader`. We can get the dimension of the system by calling `dl.input_dim`.

## Train the network 

We now want to compare two different approaches: [PSDArch](@ref) and [SymplecticAutoencoders](@ref). For this we first have to set up the networks: 

```@example linear_wave
const reduced_dim = 10

psd_arch = PSDArch(dl.input_dim, 10)
sae_arch = SymplecticAutoencoder(dl.input_dim, 10)

psd_nn = NeuralNetwork(psd_arch)
sae_nn = NeuralNetwork(sae_arch)

nothing # hide
```

Training a neural network is usually done by calling an instance of [Optimizer](@ref) in `GeometricMachineLearning`. [PSDArch](@ref) however can be solved directly by using singular value decomposition and this is done by calling [solve!](@ref). The `SymplecticAutoencoder` we train with the [AdamOptimzier](@ref) however: 

```@example linear_wave 
o = Optimizer(sae_nn, AdamOptimizer(Float64))

const n_epochs = 100

o(sae_nn, dl; n_epochs = n_epochs)
solve!(psd_nn, dl)

nothing # hide
```

## The online stage 

After having trained our neural network we can now evaluate it in the online stage of reduced complexity modeling: 

```@example linear_wave
psd_rs = HRedSys(hodeproblem(), encoder(psd_nn), decoder(psd_nn); integrator = ImplicitMidpoint())
sae_rs = HRedSys(hodeproblem(), encoder(sae_nn), decoder(sae_nn); integrator = ImplicitMidpoint())

projection_error(psd_rs)
```

```@example linear_wave 
projecction_error(sae_rs)
```

```@example 
reduction_error(psd_rs)
```

```@example 
reduction_error(sae_rs)
```


## References 
```@bibliography
Pages = []
Canonical = false

buchfink2023symplectic
peng2016symplectic
greif2019decay
```