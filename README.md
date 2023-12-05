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

Traditionally, physical properties have been encoded into the loss function (PiNN approach), but in `GeometricMachineLearning.jl` this is exclusively done through the architectures and the optimizers of the neural network, thus giving theoretical guarantees that these properties are actually preserved.

Using the package is very straightforward and is very flexible with respect to the device (CPU, CUDA, ...) and the type (Float16, Float32, Float64, ...) you want to use. The following is a simple example that should learn a sine function:
```julia
using GeometricMachineLearning
using CUDA
using Zygote
using LinearAlgebra
using Plots

model = Chain(StiefelLayer(2, 100), ResNet(100, tanh), Dense(100,2, tanh))
ps = initialparameters(CUDABackend(), Float32, model)

training_data = [CUDA.rand(2,100)*2*pi for _ in 1:1000]
function loss(ps, t)
    input = training_data[t]
    norm(sin.(input) - model(input, ps))/100
end

o = Optimizer(AdamOptimizer(), ps)

function train_one_epoch()
    for t in 1:1000
        dx = Zygote.gradient(ps -> loss(ps, t), ps)[1]
        optimization_step!(o, model, ps, dx)
    end
end

for _ in 1:1
    train_one_epoch()
end

learned_trajectories = CuArray{Float32}(0:.1:2*pi)
learned_trajectories = model(hcat(learned_trajectories, learned_trajectories)', ps)

trajectory_to_plot = Matrix{Float32}(learned_trajectories)[1,:]
plot(trajectory_to_plot)
```
The optimization of the first layer is done on the Stiefel Manifold $St(n, N)$, and the optimizer used is the manifold version of Adam (see (Brantner, 2023)).

## References
- Brantner B. Generalizing Adam To Manifolds For Efficiently Training Transformers[J]. arXiv preprint arXiv:2305.16901, 2023.
