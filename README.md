<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/JuliaGNI/GeometricMachineLearning.jl/assets/55493704/8d6d1410-b857-4e0f-8609-50e43be9a268">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/JuliaGNI/GeometricMachineLearning.jl/assets/55493704/014929d1-2297-4b2c-9359-58cadbb03a0e">
  <img alt="Shows a black logo in light color mode and a white one in dark color mode.">
</picture>


[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagni.github.io/GeometricMachineLearning.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliagni.github.io/GeometricMachineLearning.jl/latest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![PkgEval Status](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GeometricMachineLearning.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GeometricMachineLearning.html)
[![Build Status](https://github.com/JuliaGNI/GeometricMachineLearning.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/GeometricMachineLearning.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/GeometricMachineLearning.jl/branch/main/graph/badge.svg?token=CFT76RROW2)](https://codecov.io/gh/JuliaGNI/GeometricMachineLearning.jl)

`GeometricMachineLearning.jl` offers a flexible tool for designing neural networks for dynamical systems with geometric structure, such as Hamiltonian (symplectic) or Lagrangian (variational) systems.

At its core every neural network comprises three components: a neural network architecture, a loss function and an optimizer. 

Traditionally, physical properties have been encoded into the loss function (PINN approach), but in `GeometricMachineLearning.jl` this is exclusively done through the architectures and the optimizers of the neural network, thus giving theoretical guarantees that these properties are actually preserved.

Using the package is very straightforward and is very flexible with respect to the device `(CPU, CUDA, Metal, ...)` and the type `(Float16, Float32, Float64, ...)` you want to use. The following is a simple example to learn a SympNet on data coming from a pendulum:
```julia
using GeometricMachineLearning
using CUDA # Metal
using Plots

include("scripts/pendulum.jl")

type = Float32 # Float16 etc.
# get data 
qp_data = GeometricMachineLearning.apply_toNT(a -> CuArray(type.(a)), pendulum_data((q=[0.], p=[1.]); timespan=(0.,100.)))
# call the DataLoader
dl = DataLoader(qp_data)

# call the SympNet architecture
gsympnet = GSympNet(dl)

# specify the backend
backend = CUDABackend()

# initialize the network (i.e. the parameters of the network)
g_nn = NeuralNetwork(gsympnet, backend, type)

# call the optimizer
g_opt = Optimizer(AdamOptimizer(), g_nn)

const nepochs = 300
const batch_size = 100

# train the network
g_loss_array = g_opt(g_nn, dl, Batch(batch_size), nepochs)

# plot the result
ics = (q=qp_data.q[:,1], p=qp_data.p[:,1])
const steps_to_plot = 200
g_trajectory = Iterate_Sympnet(g_nn, ics; n_points = steps_to_plot)
p2 = plot(qp_data.q'[1:steps_to_plot], qp_data.p'[1:steps_to_plot], label="training data")
plot!(p2, g_trajectory.q', g_trajectory.p', label="G Sympnet")
```
More examples like this can be found in the docs.

## References
- Brantner B. Generalizing Adam To Manifolds For Efficiently Training Transformers. arXiv preprint arXiv:2305.16901, 2023.
- Brantner B., Kraus M. Symplectic Autoencoders for Model Reduction of Hamiltonian Systems. arXiv preprint arXiv:2312.10004, 2023.
- Brantner B., Romemont G., Kraus M., Li Z. Structure-Preserving Transformers for Learning Parametrized Hamiltonian Systems. arXiv preprint arXiv:2312.11166, 2023.


## Development

We are using git hooks, e.g., to enforce that all tests pass before pushing.
In order to activate these hooks, the following command must be executed once:
```
git config core.hooksPath .githooks
```
