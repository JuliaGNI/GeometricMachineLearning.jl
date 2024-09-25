# Hamiltonian Neural Network

In this tutorial we build a *Hamiltonian neural network*. We first need vector field data:

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
hnn_arch = Chain(Dense(2, intermediate_dim, tanh), Dense(intermediate_dim, intermediate_dim, tanh), Linear(intermediate_dim, 1))
hnn = NeuralNetwork(hnn_arch)
nothing # hide
```

Next we define the loss function

```@example hnn
struct HNNLoss <: NetworkLoss end
function (loss::HNNLoss)(model::Chain, ps::Tuple, input::AT, output::AT) where {T, AT <: AbstractArray{T}}
    vf = 𝕁(gradient(input -> sum(model(input, ps)), input)[1])
    norm(input - output)
end
loss = HNNLoss()
nothing # hide
```

We can now train the network

```@example hnn
batch = Batch(10)
n_epochs = 100
o = Optimizer(AdamOptimizer(Float64), hnn)
loss_array = o(hnn, dl, batch, n_epochs, loss)
```