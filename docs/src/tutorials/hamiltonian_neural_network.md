# Hamiltonian Neural Network(@id hnn_tutorial)

In this tutorial we build a [Hamiltonian neural network](@ref hnn_architecture). 

## Training a HNN Based on VectorField Data

We first train a HNN [based on vector field data](@ref "HNN Loss for Vector Field Data"):

```@example hnn
using GeometricMachineLearning # hide
using GeometricMachineLearning: QPT
using LinearAlgebra: norm
using Zygote: gradient

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

```julia
batch = Batch(10)
n_epochs = 100
o = Optimizer(AdamOptimizer(Float64), hnn)
loss_array = o(hnn, dl, batch, n_epochs, loss)
```

!!! info
   Usually we use [`Zygote`](https://github.com/FluxML/Zygote.jl) for computing derivatives in `GeometricMachineLearning`, but as the [`Zygote` documentation](https://fluxml.ai/Zygote.jl/dev/limitations/#Second-derivatives-1) itself points out: "Often using a different AD system over Zygote is a better solution [for computing second-order derivatives]." For this reason we compute the loss of the HNN with [`SymbolicNeuralNetworks`](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl) and optionally also its gradient.

## Training a HNN Based on Phase Space Data