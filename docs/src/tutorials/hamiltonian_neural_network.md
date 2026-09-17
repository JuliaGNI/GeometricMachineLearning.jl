# [Hamiltonian Neural Network](@id hnn_tutorial)

In this tutorial we build a [Hamiltonian neural network](@ref hnn_architecture). 

## Training a HNN Based on VectorField Data

We first train a HNN [based on vector field data](@ref "HNN Loss for Vector Field Data"):

```@example hnn
using GeometricMachineLearning # hide
using GeometricMachineLearning: QPT
using LinearAlgebra: norm
using Zygote: gradient
import Random # hide
Random.seed!(1234) # hide

𝕁 = PoissonTensor(2)
vf(z) = 𝕁 * z
domain = [[q, p] for q in -1:.1:1 for p in -1:.1:1]
vf_data = vf.(domain)
domain_matrix = hcat(domain...)
vf_matrix = hcat(vf_data...)
dl = DataLoader(domain_matrix, vf_matrix)
nothing # hide
```

We then build the neural network:

```@example hnn
const intermediate_dim = 5
hnn_arch = StandardHamiltonianArchitecture(2, intermediate_dim)
hnn = NeuralNetwork(hnn_arch)
nothing # hide
```

Next we define the loss function

```@example hnn
loss = HNNLoss(hnn_arch)
nothing # hide
```

We can now train the network

```@example hnn
batch = Batch(10)
n_epochs = 100
o = Optimizer(Adam(Float64), hnn)
loss_array = o(hnn, dl, batch, n_epochs, loss)
using CairoMakie # hide
lines(loss_array) # hide
```

!!! info
    Usually we use [`Zygote`](https://github.com/FluxML/Zygote.jl) for computing derivatives in `GeometricMachineLearning`, but as the [`Zygote` documentation](https://fluxml.ai/Zygote.jl/dev/limitations/#Second-derivatives-1) itself points out: "Often using a different AD system over Zygote is a better solution [for computing second-order derivatives]." For this reason we compute the loss of the HNN with [`SymbolicNeuralNetworks`](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl) and optionally also its gradient.

## Training a HNN Based on Phase Space Data

We now train a HNN on the same system [based on phase space data](@ref "HNN Loss for Phase Space Data"). The data are not vector fields, but pairs of points a fixed timestep apart. We produce such pairs by applying the exact flow of `vf`, which is ``\exp(\Delta{}t\mathbb{J})``, to the points of the domain:

```@example hnn
const Δt = .1
next_matrix = exp(Δt * Matrix(𝕁)) * domain_matrix
dl_pairs = DataLoader(domain_matrix, next_matrix)
nothing # hide
```

We start from a new network of the architecture we built above:

```@example hnn
hnn_pairs = NeuralNetwork(hnn_arch)
nothing # hide
```

The loss needs the timestep, because the finite difference it compares the vector field against does:

```@example hnn
loss_pairs = SymplecticEulerLoss(hnn_arch, Δt)
nothing # hide
```

[`GeometricMachineLearning.SymplecticEulerLoss`](@ref) defaults to the `:A` variant, which evaluates the vector field at ``(q^{(t+1)}, p^{(t)})``. Passing `variant = :B` evaluates it at ``(q^{(t)}, p^{(t+1)})`` instead, which is the variant the [loss formula](@ref "HNN Loss for Phase Space Data") is written for.

We can now train the network:

```@example hnn
o_pairs = Optimizer(Adam(Float64), hnn_pairs)
loss_array_pairs = o_pairs(hnn_pairs, dl_pairs, batch, n_epochs, loss_pairs)
lines(loss_array_pairs) # hide
```
