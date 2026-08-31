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

The optimizer methods themselves — `GradientMethod`, `MomentumMethod`, `Adam`, the manifold types they act on, and the caches, global sections and retractions that go with them — come from [`GeometricOptimizers.jl`](https://github.com/JuliaGNI/GeometricOptimizers.jl) and are re-exported here. `GeometricMachineLearning.jl` supplies the part that is specific to neural networks: the architectures, the layers, and walking the parameter tree of a network during training.

Using the package is very straightforward and is very flexible with respect to the device `(CPU, CUDA, Metal, ...)` and the type `(Float16, Float32, Float64, ...)` you want to use. The following is a simple example to learn a SympNet on data coming from a pendulum:
```julia
using GeometricMachineLearning
using CUDA # Metal
using CairoMakie

include("scripts/pendulum.jl")

type = Float32 # Float16 etc.
# get data 
qp_data = map(a -> CuArray(type.(a)), pendulum_data((q=[0.], p=[1.]); timespan=(0.,100.)))
# call the DataLoader
dl = DataLoader(qp_data)

# call the SympNet architecture
gsympnet = GSympNet(dl)

# specify the backend
backend = CUDABackend()

# initialize the network (i.e. the parameters of the network)
g_nn = NeuralNetwork(gsympnet, backend, type)

# call the optimizer: the method comes first, the step size is given separately
g_opt = Optimizer(Adam(type), g_nn; step_size = 1e-3)

const nepochs = 300
const batch_size = 100

# train the network
g_loss_array = g_opt(g_nn, dl, Batch(batch_size), nepochs)

# plot the result
ics = (q=qp_data.q[:,1], p=qp_data.p[:,1])
const steps_to_plot = 200
g_trajectory = iterate(g_nn, ics; n_points = steps_to_plot)
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "q", ylabel = "p")
lines!(ax, vec(qp_data.q')[1:steps_to_plot], vec(qp_data.p')[1:steps_to_plot]; label = "training data")
lines!(ax, vec(g_trajectory.q'), vec(g_trajectory.p'); label = "G Sympnet")
axislegend(ax)
fig
```
More examples like this can be found in the docs.

## References
- Brantner B. Generalizing Adam To Manifolds For Efficiently Training Transformers. arXiv preprint arXiv:2305.16901, 2023.
- Brantner B., Kraus M. Symplectic Autoencoders for Model Reduction of Hamiltonian Systems. arXiv preprint arXiv:2312.10004, 2023.
- Brantner B., Romemont G., Kraus M., Li Z. Structure-Preserving Transformers for Learning Parametrized Hamiltonian Systems. arXiv preprint arXiv:2312.11166, 2023.


## Development

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
