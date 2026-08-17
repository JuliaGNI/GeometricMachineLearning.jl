
# Simple Julia Models for Learning Hamiltonian Dynamics

> Legacy. These scripts predate the current `GeometricMachineLearning` API and are kept for
> reference, not run. Six of them include `../../scripts/data.jl`, which does not exist, so those
> six do not get as far as the network.

**hnn_simple.jl** implements a simple handmade neural network including the learning steps. Here [**Zygote**](https://github.com/FluxML/Zygote.jl) or [**ForwardDiff**](https://github.com/JuliaDiff/ForwardDiff.jl) can be used to compute the gradient of the loss with respect to the weights of the NN.

**hnn_mt.jl**  uses [**ModelingToolkit**](https://github.com/SciML/ModelingToolkit.jl) for the computation of all gradients. The relevant functions (est -> estimate of the Hamiltonian, field -> estimate of the vector field, loss -> the loss function, and stepfun -> function that performs one gradient step) are generated with **ModelingToolkit**. The script that generates them (**generate_hnn_mt.jl**) as well as the generated files are in the sibling directory **mtk**. 

**hnn_flux.jl** implements the network using [**Flux**](https://github.com/FluxML/Flux.jl) but with a custom training function.

**data.jl** contains functions that generate the data set. The scripts look for it at
`../../scripts/data.jl`, where there is no such file.
**networks.jl**, in this directory, builds the network that estimates the Hamiltonian and computes the vector field with **Zygote**.
**../../scripts/plots.jl** performs a few plots that demonstrate energy conservation.
