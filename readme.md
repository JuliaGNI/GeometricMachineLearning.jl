# Simple model for learning Hamiltonians 

**utils.jl** builds the network that estimates the Hamiltonian and computes the field with **Zygote** (https://github.com/FluxML/Zygote.jl). **hnn.jl** performs the numerics, i.e. learning steps with the functions from **utils.jl**. Here **Zygote** or **ForwardDiff** (https://github.com/JuliaDiff/ForwardDiff.jl) can be used to compute the gradient of the loss with respect to the weights of the NN.  

**hnn_mt.jl** produces similar results to **hnn.jl**, but uses **ModelingToolkit** (https://github.com/SciML/ModelingToolkit.jl) instead. The relevant functions generated with **ModelingToolkit**, and the program that generates them, are in the directory **mt_fun**. 


**data.jl** contains a function that generates the data set, and **plots.jl** performs a few plots that demonstrate energy conservation.  




## Things left to do

- [] perhabs generalize the symplectic matrix to arbitrary dimensions (instead of [0 1; -1 0])



