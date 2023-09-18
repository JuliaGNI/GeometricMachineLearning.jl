# Symplectic Autoencoder 

Symplectic Autoencoders are a type of neural network suitable for treating Hamiltonian parametrized PDEs with slowly decaying Kolmogorov $n$-width. It is based on **proper symplectic decomposition** (PSD) and symplectic neural networks (SympNets).

## Hamiltonian Model Order Reduction 

Hamiltonian PDEs are partial differential equations that, like its ODE counterpart, have a Hamiltonian associated with it. An example of this is the **linear wave equation** (see (Buckfink et al, 2023)) with Hamiltonian 

```math
\mathcal{H}(q, p; \mu) := \frac{1}{2}\int_\Omega\mu^2(\partial_\xi{}q(t,\xi;\mu))^2 + p(t,\xi;\mu)^2d\xi.
```

The PDE for to this Hamiltonian can be obtained similarly as in the ODE case:

```math
\partial_t{}q(t,\xi;\mu) = \frac{\delta{}\mathcal{H}}{\delta{}p} = p(t,\xi;\mu), \quad \partial_t{}p(t,\xi;\mu) = -\frac{\delta{}\mathcal{H}}{\delta{}q} = \mu^2\partial_{\xi{}\xi}q(t,\xi;\mu)
```

## Symplectic Solution Manifold 

As with regular parametric PDEs, we also associate a solution manifold with Hamiltonian PDEs. This is a finite-dimensional manifold, on which the dynamics can be described through a Hamiltonian ODE. 
I NEED A PROOF OR SOME EXPLANATION FOR THIS!


## Workflow for Symplectic ROM

As with any other [reduced order modeling technique](autoencoder.md) we first discretize the PDE. This should be done with a structure-preserving scheme, thus yielding a (high-dimensional) Hamiltonian ODE as a result. Discretizing the wave equation above with finite differences yields a Hamiltonian system: 

```math
\mathcal{H}_\mathrm{discr}(z(t;\mu);\mu) := \frac{1}{2}x(t;\mu)^T\begin{bmatrix}  -\mu^2D_{\xi{}\xi} & \mathbb{O} \\ \mathbb{O} & \mathbb{I}  \end{bmatrix} x(t;\mu).
```

In Hamiltonian reduced order modelling we try to find a symplectic submanifold of the solution space[^1] that captures the dynamics of the full system as well as possible.

[^1]: The submanifold is: $\tilde{\mathcal{M}} = \{\Psi^\mathrm{dec}(z_r)\in\mathbb{R}^{2N}:u_r\in\mathrm{R}^{2n}\}$ where $z_r$ is the reduced state of the system. 

Similar to the regular PDE case we again build an encoder $\Psi^\mathrm{enc}$ and a decoder $\Psi^\mathrm{dec}$; but now both these mappings are required to be symplectic!

Concretely this means: 
1. The encoder is a mapping from a high-dimensional symplectic space to a low-dimensional symplectic space, i.e. $\Psi^\mathrm{enc}:\mathbb{R}^{2N}\to\mathbb{R}^{2n}$ such that $\nabla\Psi^\mathrm{enc}\mathbb{J}_{2N}(\nabla\Psi^\mathrm{enc})^T = \mathbb{J}_{2n}$.
2. The decoder is a mapping from a low-dimensional symplectic space to a high-dimensional symplectic space, i.e. $\Psi^\mathrm{dec}:\mathbb{R}^{2n}\to\mathbb{R}^{2N}$ such that $(\nabla\Psi^\mathrm{dec})^T\mathbb{J}_{2N}\nabla\Psi^\mathrm{dec} = \mathbb{J}_{2n}$.


## References 
- Buchfink, Patrick, Silke Glas, and Bernard Haasdonk. "Symplectic model reduction of Hamiltonian systems on nonlinear manifolds and approximation with weakly symplectic autoencoder." SIAM Journal on Scientific Computing 45.2 (2023): A289-A311.