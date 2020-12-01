# Simple model for learning Hamiltonians 

**utils.jl** builds the network that estimates the Hamiltonian and computes the field with [**Zygote**](https://github.com/FluxML/Zygote.jl). **hnn_simple.jl** performs the numerics, i.e. learning steps with the functions from **utils.jl**. Here **Zygote** or [**ForwardDiff**](https://github.com/JuliaDiff/ForwardDiff.jl) can be used to compute the gradient of the loss with respect to the weights of the NN.

**hnn_mt.jl** produces similar results to **hnn_simple.jl**, but uses [**ModelingToolkit**](https://github.com/SciML/ModelingToolkit.jl) instead. The relevant functions (est -> estimate of the Hamiltonian, field -> estimate of the vector field, loss, stepfun -> i.e. function that performs one gradient step) generated with **ModelingToolkit**, and the program that generates them (**hnn_mt.jl**), are in the directory **mt_fun**. 

**hnn_flux.jl** implements the network using [**Flux**](https://github.com/FluxML/Flux.jl).


**data.jl** contains a function that generates the data set, and **plots.jl** performs a few plots that demonstrate energy conservation.  




## Things left to do

- Generalize the symplectic matrix to arbitrary dimensions (instead of [0 1; -1 0]).

