# Geometric Machine Learning

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagni.github.io/GeometricMachineLearning.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliagni.github.io/GeometricMachineLearning.jl/latest/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![PkgEval Status](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GeometricMachineLearning.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GeometricMachineLearning.html)
[![CI](https://github.com/JuliaGNI/GeometricMachineLearning.jl/workflows/CI/badge.svg)](https://github.com/JuliaGNI/GeometricMachineLearning.jl/actions?query=workflow:CI)
[![Codecov Status](https://codecov.io/gh/JuliaGNI/GeometricMachineLearning.jl/branch/main/graph/badge.svg?token=CFT76RROW2)](https://codecov.io/gh/JuliaGNI/GeometricMachineLearning.jl)

`GeometricMachineLearning.jl` offers a flexible tool for designing neural networks for dynamical systems with geometric structure, such as Hamiltonian (symplectic) or Lagrangian (variational) systems.

At its core every neural network comprises three components: a neural network architecture, a loss function and an optimizer. 

Traditionally, physical properties have been encoded into the loss function (PiNN approach), but in `GeometricMachineLearning.jl` this is exclusively done through the architectures and the optimizers of the neural network, thus giving theoretical guarentees that these properties are actually preserved.
