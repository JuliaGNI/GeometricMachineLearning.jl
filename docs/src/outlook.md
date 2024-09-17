# Conclusion and Outlook

In this dissertation it was shown how neural networks can be imbued with structure to improve their approximation capabilities when applied to physical systems. Here we again summarize the novelties of this work and give an outlook for future work.

## Structure-Preserving Reduced Order Modeling of Hamiltonian Systems

A central part of this dissertation was the development of [symplectic autoencoders](@ref "The Symplectic Autoencoder"). These were used to reduce a 400-dimensional Hamiltonian system to a two-dimensional one[^1]:

[^1]: ``\bar{H} = H\circ\Psi^\mathrm{dec}_{\theta_2}:\mathbb{R}^2\to\mathbb{R}`` here refers to the *induced Hamiltonian on the reduced space*. 

```math
(\mathbb{R}^{400}, H) \xRightarrow{\mathrm{SAE}} (\mathbb{R}^2, \bar{H}).
```

For this case we observed dramatic speed-ups [of up to a factor of 1000](@ref "Symplectic Autoencoders and the Toda Lattice").

## Structure-Preserving Neural Network-Based Integrators

## Structure-Preserving Optimization Schemes

## Outlook