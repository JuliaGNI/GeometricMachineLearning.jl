# Conclusion and Outlook

In this dissertation it was shown how neural networks can be imbued with structure to improve their approximation capabilities when applied to physical systems. Here we again summarize the novelties of this work and give an outlook for how it can be expanded in the future.

## Reduced Order Modeling as Motivation

Most of the work presented in this dissertation is motivated by [data-driven reduced order modeling](@ref "Basic Concepts of Reduced Order Modeling"). This is the discipline of building a low-dimensional surrogate model from data that come from a high-dimensional full order model. Both the low-dimensional surrogate model and the high-dimensional full order model are described by a differential equation. When we talk about *structure-preserving reduced order modeling* we mean that the equation on the low-dimensional space [shares features with the equation on the high-dimensional space](@ref "Hamiltonian Model Order Reduction"). In this work these properties were mainly for the vector field to be [symplectic](@ref "Symplectic Systems") or [divergence-free](@ref "Divergence-Free Vector Fields"). A typical reduced order modeling framework is further divided into two phases:
1. in the *offline phase* we find the low-dimensional surrogate model (reduced representation) and
2. in the *online phase* we solve the equations on the reduced space.

For the offline phase we proposed [symplectic autoencoders](@ref "The Symplectic Autoencoder") and for the online phase we proposed [volume-preserving transformers](@ref "Volume-Preserving Transformer") and [linear symplectic transformers](@ref "Linear Symplectic Transformer").

## Structure-Preserving Reduced Order Modeling of Hamiltonian Systems - The Offline Phase

A central part of this dissertation was the development of [symplectic autoencoders](@ref "The Symplectic Autoencoder") [brantner2023symplectic](@cite). We used these to reduce a 400-dimensional Hamiltonian system to a two-dimensional one[^1]:

[^1]: ``\bar{H} = H\circ\Psi^\mathrm{dec}_{\theta_2}:\mathbb{R}^2\to\mathbb{R}`` here refers to the *induced Hamiltonian on the reduced space*. 

```math
(\mathbb{R}^{400}, H) \xRightarrow{\mathrm{SAE}} (\mathbb{R}^2, \bar{H}).
```

For this case we observed dramatic speed-ups of up to a factor 1000 when [symplectic autoencoder was coupled with a transformer in the online phase](@ref "Symplectic Autoencoders and the Toda Lattice").

Symplectic autoencoders have the property that they induce a Hamiltonian system on the reduced space. This distinguishes them from so-called *weakly symplectic autoencoders* [buchfink2023symplectic, yildiz2024data](@cite) that only approximately obtain a Hamiltonian system on a restricted domain via a PINN [raissi2019physics](@cite) approach.

## Structure-Preserving Neural Network-Based Integrators - The Online Phase

For the online phase of reduced order modeling we developed new neural network architectures based on the transformer [vaswani2017attention](@cite) which is a neural network architecture that is extensively used in other fields of neural network research such as natural language processing[^2]. We used transformers to build an equivalent of structure-preserving multi-step methods [hairer2006geometric](@cite).

[^2]: The T in ChatGPT [achiam2023gpt](@cite) stands for *transformer*.

One of these architectures is the [volume-preserving transfomrer](@ref "Volume-Preserving Transformer") [brantner2024volume](@cite) and ...


## Structure-Preserving Optimization Schemes

Training a symplectic autoencoder requires ...

## Outlook

For future work we foresee ...