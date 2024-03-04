# Symplectic Autoencoder 

Symplectic Autoencoders are a type of neural network suitable for treating Hamiltonian parametrized PDEs with slowly decaying Kolmogorov $n$-width. It is based on **proper symplectic decomposition** (PSD) and symplectic neural networks (SympNets).

## Hamiltonian Model Order Reduction 

Hamiltonian PDEs are partial differential equations that, like its ODE counterpart, have a Hamiltonian associated with it. An example of this is the **linear wave equation** (see [buchfink2023symplectic](@cite)) with Hamiltonian 

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

If these two maps are constrained to linear maps, then one can easily find good solutions with **proper symplectic decomposition** (PSD).

## Proper Symplectic Decomposition

For PSD the two mappings $\Psi^\mathrm{enc}$ and $\Psi^\mathrm{dec}$ are constrained to be linear, orthonormal (i.e. $\Psi^T\Psi = \mathbb{I}$) and symplectic. The easiest way to enforce this is through the so-called **cotangent lift**: 

```math
\Psi_\mathrm{CL} = 
\begin{bmatrix} \Phi & \mathbb{O} \\ \mathbb{O} & \Phi \end{bmatrix},
```
and $\Phi\in{}St(n,N)\subset\mathbb{R}^{N\times{}n}$, i.e. is an element of the [Stiefel manifold](../manifolds/stiefel_manifold.md). If the [snapshot matrix](../data_loader/snapshot_matrix.md) is of the form: 

```math
M = \left[\begin{array}{c:c:c:c}
\hat{q}_1(t_0) &  \hat{q}_1(t_1) & \quad\ldots\quad & \hat{q}_1(t_f) \\
\hat{q}_2(t_0) &  \hat{q}_2(t_1) & \ldots & \hat{q}_2(t_f) \\
\ldots & \ldots & \ldots & \ldots \\
\hat{q}_N(t_0) &  \hat{q}_N(t_1) & \ldots & \hat{q}_N(t_f) \\
\hat{p}_1(t_0) & \hat{p}_1(t_1) & \ldots & \hat{p}_1(t_f) \\
\hat{p}_2(t_0) &  \hat{p}_2(t_1) & \ldots & \hat{p}_2(t_f) \\
\ldots &  \ldots & \ldots & \ldots \\
\hat{p}_{N}(t_0) &  \hat{p}_{N}(t_1) & \ldots & \hat{p}_{N}(t_f) \\
\end{array}\right],
```

then $\Phi$ can be computed in a very straight-forward manner: 
1. Rearrange the rows of the matrix $M$ such that we end up with a $N\times2(f+1)$ matrix: $\hat{M} := [M_q, M_p]$.
2. Perform SVD: $\hat{M} = U\Sigma{}V^T$; set $\Phi\gets{}U\mathtt{[:,1:n]}$.

For details on the cotangent lift (and other methods for linear symplectic model reduction) consult [peng2016symplectic](@cite).

## Symplectic Autoencoders

PSD suffers from the similar shortcomings as regular POD: it is a linear map and the approximation space $\tilde{\mathcal{M}}= \{\Psi^\mathrm{dec}(z_r)\in\mathbb{R}^{2N}:u_r\in\mathrm{R}^{2n}\}$ is strictly linear. For problems with slowly-decaying [Kolmogorov $n$-width](kolmogorov_n_width.md) this leads to very poor approximations.  

In order to overcome this difficulty we use neural networks, more specifically [SympNets](../architectures/sympnet.md), together with cotangent lift-like matrices. The resulting architecture, symplectic autoencoders, are demonstrated in the following image: 

```@example
import Images, Plots # hide
if Main.output_type == :html_output # hide
    HTML("""<object type="image/svg+xml" class="display-light-only" data=$(joinpath(Main.buildpath, "../tikz/symplectic_autoencoder.png"))></object>""") # hide
else # hide
    Plots.plot(Images.load("../tikz/symplectic_autoencoder.png"), axis=([], false)) # hide
end # hide
```

```@example
if Main.output_type == :html_output # hide
    HTML("""<object type="image/svg+xml" class="display-dark-only" data=$(joinpath(Main.buildpath, "../tikz/symplectic_autoencoder_dark.png"))></object>""") # hide
end # hide
```

So we alternate between SympNet and PSD layers. Because all the PSD layers are based on matrices $\Phi\in{}St(n,N)$ we have to [optimize on the Stiefel manifold](../Optimizer.md).


## References 

```@bibliography
Pages = []
Canonical = false

buchfink2023symplectic
peng2016symplectic
```