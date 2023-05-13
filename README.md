# Geometric Machine Learning

`GeometricMachineLearning.jl` offers a flexible tool for designing neural networks for dynamical systems with geometric structure, such as Hamiltonian (symplectic) or Lagrangian (variational) systems.

At its core every neural network comprises three components: a neural network architecture, a loss function and an optimizer. 

Traditionally, physical properties have been encoded into the loss function (PiNN approach), but in `GeometricMachineLearning.jl` this is exclusively done through the architectures and the optimizers of the neural network, thus giving theoretical guarentees that these properties are actually preserved.
