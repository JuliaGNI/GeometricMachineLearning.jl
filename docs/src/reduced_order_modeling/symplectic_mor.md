# Hamiltonian Model Order Reduction 

Hamiltonian PDEs are partial differential equations that, like its ODE counterpart, have a Hamiltonian associated with it. The linear wave equation can be written as such a Hamiltonian PDE with 

```math
\mathcal{H}(q, p; \mu) := \frac{1}{2}\int_\Omega\mu^2(\partial_\xi{}q(t,\xi;\mu))^2 + p(t,\xi;\mu)^2d\xi.
```

Note that in contrast to the ODE case where the Hamiltonian is a function ``H(\cdot; \mu):\mathbb{R}^{2d}\to\mathbb{R},`` we now have a functional ``\mathcal{H}(\cdot, \cdot; \mu):\mathcal{C}^\infty(\mathcal{D})\times\mathcal{C}^\infty(\mathcal{D})\to\mathbb{R}.`` The PDE for to this Hamiltonian can be obtained similarly as in the ODE case:

```math
\partial_t{}q(t,\xi;\mu) = \frac{\delta{}\mathcal{H}}{\delta{}p} = p(t,\xi;\mu), \quad \partial_t{}p(t,\xi;\mu) = -\frac{\delta{}\mathcal{H}}{\delta{}q} = \mu^2\partial_{\xi{}\xi}q(t,\xi;\mu)
```

Neglecting the Hamiltonian structure of a system can have grave consequences on the performance of the reduced order model [peng2016symplectic, buchfink2023symplectic, tyranowski2023symplectic](@cite) which is why all algorithms in `GeometricMachineLearning` designed for producing reduced order models respect the structure of the system.

## The Symplectic Solution Manifold 

As with regular parametric PDEs, we also associate a solution manifold with Hamiltonian PDEs. This is a finite-dimensional manifold, on which the dynamics can be described through a Hamiltonian ODE. The reduced system, with which we approximate this symplectic solution manifold, is a low dimensional symplectic vector space ``\mathbb{R}^{2n}`` together with a reduction ``\mathcal{P}`` and a reconstruction ``\mathcal{R}.`` If we now take an initial condition on the solution manifold ``\hat{u}_0\in\mathcal{M} \approx \mathcal{R}(\mathbb{R}^{2n})`` and project it to the reduced space with ``\mathcal{P}``, we get ``u = \mathcal{P}(\hat{u}_0).`` We can now integrate it on the reduced space via the induced differential equation, which is of canonical Hamiltonian form, and obtain an orbit ``u(t)`` which can then be mapped back to an orbit on the solution manifold[^4] via ``\mathcal{R}.`` The resulting orbit ``\mathcal{R}(u(t))`` is ideally the unique orbit on the full order model ``\hat{u}(t)\in\mathcal{M}``.

[^4]: To be precise, an *approximation of the solution manifold* ``\mathcal{R}(\mathbb{R}^{2n})``, because we are not able to find the solution manifold exactly in practice. 


For Hamiltonian model order reduction we additionally require that the reduction ``\mathcal{P}`` satisfies

```math
    \nabla_z\mathcal{P}\mathbb{J}_{2N}(\nabla_z\mathcal{P})^T = \mathbb{J}_{2N} \text{ for $z\in\mathbb{R}^{2N}$}
```

and the reconstruction ``\mathcal{R}`` satisfies[^5]

[^5]: We should note that satisfying this *symplecticity condition* is much more important for the reconstruction than for the reduction. We are not entirely sure if the condition on the reduction is really necessary. In [buchfink2023symplectic](@cite) it is ignored.

```math
    (\nabla_z\mathcal{R})^T\mathbb{J}_{2N}\nabla_z\mathcal{R} = \mathbb{J}_{2n}.
```

With this we have

```@eval
Main.theorem(raw"A Hamiltonian system on the reduced space ``(\mathbb{R}^{2n}, \mathbb{J}_{2n}^T)`` is equivalent to a *non-canonical symplectic system* on ``(\mathcal{M}, \mathbb{J}_{2N}^T|_\mathcal{M})`` where 
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathcal{M} = \mathcal{R}(\mathbb{R}^{2n})
" * Main.indentation * raw"```
" * Main.indentation * raw"is an approximation to the solution manifold.")
```

For the proof we use the fact that ``\mathcal{M} = \mathcal{R}(\mathbb{R}^{2n})`` is a manifold whose coordinate chart is the [*local inverse* of ``\mathcal{R}``](@ref "The Immersion Theorem") which we will call ``\psi``, i.e. around a point ``y\in\mathcal{M}`` we have ``\mathcal{R}\circ\psi(y) = y``.
```@eval
Main.proof(raw"Note that the tangent space at ``y = \mathcal{R}(z)`` to ``\mathcal{M}`` is:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    T_y\mathcal{M} = \{(\nabla_z\mathcal{R})v: v\in\mathbb{R}^{2n}\}.
" * Main.indentation * raw"```
" * Main.indentation * raw"The mapping 
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathcal{M} \to T\mathcal{M}, y \mapsto (\nabla_z\mathcal{R})\mathbb{J}_{2n}\nabla_zH
" * Main.indentation * raw"```
" * Main.indentation * raw"is clearly a vector field. We now prove that it is symplectic and equal to ``\mathbb{J}_{2N}\nabla_y(H\circ\psi).`` For this first note that we have ``\mathbb{I} = (\nabla_z\mathcal{R})^+\nabla_z\mathcal{R} = (\nabla_{\mathcal{R}(z)}\psi)\nabla_z\mathcal{R}`` and that the pseudoinverse is unique. We then have:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathbb{J}_{2N}\nabla_yH\circ\psi = \mathbb{J}_{2N}(\nabla_y\psi)^T\nabla_{\psi(y)}H = \mathbb{J}_{2N}\left((\nabla_{\psi(y)}\mathcal{R})^+\right)^T\nabla_{\psi(y)}H = (\nabla_{\psi(y)}\mathcal{R})\mathbb{J}_{2n}\nabla_{\psi(y)}H,
" * Main.indentation * raw"```
" * Main.indentation * raw"which proves that every Hamiltonian system on ``\mathbb{R}^{2n}`` induces a Hamiltonian system on ``\mathcal{M}``. Conversely we built the manifold ``\mathcal{M}`` based on the flow of a high-dimensional Hamiltonian system. We can thus approximate the high-dimensional Hamiltonian system with a low-dimensional Hamiltonian system on ``\mathbb{R}^{2n}``.")
```

This theorem serves as the basis for Hamiltonian model order reduction via [proper symplectic decomposition](@ref "Proper Symplectic Decomposition") and [symplectic autoencoders](@ref "Symplectic Autoencoders").

## Proper Symplectic Decomposition

For proper symplectic decomposition (PSD) the reduction ``\mathcal{P}`` and the reconstruction ``\mathcal{R}`` are constrained to be linear, orthonormal and symplectic. Note that these first two properties are shared with [POD](@ref "Proper Orthogonal Decomposition"). The easiest way[^1] to enforce this is through the so-called "cotangent lift" [peng2016symplectic](@cite): 

[^1]: The original PSD paper [peng2016symplectic](@cite) proposes another approach to build linear reductions and reconstructions with the so-called "complex SVD." In practice this only brings minor advantages over the cotangent lift however [tyranowski2023symplectic](@cite).

```math
\mathcal{R} \equiv \Psi_\mathrm{CL} = \begin{bmatrix} \Phi & \mathbb{O} \\ \mathbb{O} & \Phi \end{bmatrix} \text{ where $\Psi_\mathrm{CL}\in{}St(n,N)\subset\mathbb{R}^{N\times{}n}$},
```
i.e. is an element of the [Stiefel manifold](@ref "The Stiefel Manifold"). If the [snapshot matrix](@ref "Snapshot Matrix") is of the form: 

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

then ``\Phi_\mathrm{CL}`` can be computed in a very straight-forward manner:

```@eval
Main.theorem(raw"The ideal cotangent lift ``\Psi_\mathrm{CL}`` for the snapshot matrix of form
" * Main.indentation * raw"```math
" * Main.indentation * raw"    M = \begin{bmatrix} M_q \\ M_p \end{bmatrix},
" * Main.indentation * raw"```
" * Main.indentation * raw"i.e. the cotangent lift that minimizes the projection error, can be obtained the following way:
" * Main.indentation * raw"1. Rearrange the rows of the matrix ``M`` such that we end up with a ``N\times(2\mathtt{nts})`` matrix: ``\hat{M} := [M_q, M_p]``.
" * Main.indentation * raw"2. Perform SVD: ``\hat{M} = V\Sigma{}U^T.`` 
" * Main.indentation * raw"3. Set ``\Phi\gets{}U\mathtt{[:,1:n]}.``
" * Main.indentation * raw"``\Psi_\mathrm{CL}`` is then built based on this ``\Phi``.")
```

For details on the cotangent lift (and other methods for linear symplectic model reduction) consult [peng2016symplectic](@cite). In `GeometricMachineLearning` we use the function [`solve!`](@ref) for this task.

## Symplectic Autoencoders

Symplectic Autoencoders are a type of neural network suitable for treating Hamiltonian parametrized PDEs with slowly decaying [Kolmogorov ``n``-width](@ref). It is based on PSD and [symplectic neural networks (SympNets)](@ref "SympNet Architecture").

PSD suffers from the similar shortcomings as regular POD: it is a linear map and the approximation space ``\tilde{\mathcal{M}}= \{\Psi^\mathrm{dec}(z_r)\in\mathbb{R}^{2N}:z_r\in\mathrm{R}^{2n}\}`` is therefore also linear. For problems with slowly-decaying [Kolmogorov ``n``-width](@ref) this leads to very poor approximations.  

In order to overcome this difficulty we use neural networks, more specifically [SympNets](@ref "SympNet Architecture"), together with cotangent lift-like matrices. The resulting architecture, symplectic autoencoders, are discussed in the [dedicated section on neural network architectures](@ref "The Symplectic Autoencoder").

## Workflow for Symplectic ROM

As with any other [reduced order modeling technique](@ref "General Workflow") we first discretize the PDE. This should be done with a structure-preserving scheme, thus yielding a (high-dimensional) Hamiltonian ODE as a result. Going back to the example of the linear wave equation, we can discretize this equation with finite differences to obtain a Hamiltonian ODE: 

```math
\mathcal{H}_\mathrm{discr}(z(t;\mu);\mu) := \frac{1}{2}x(t;\mu)^T\begin{bmatrix}  -\mu^2D_{\xi{}\xi} & \mathbb{O} \\ \mathbb{O} & \mathbb{I}  \end{bmatrix} x(t;\mu).
```

In Hamiltonian reduced order modelling we try to find a symplectic submanifold in the solution space[^2] that captures the dynamics of the full system as well as possible.

[^2]: The submanifold, that approximates the solution manifold, is ``\tilde{\mathcal{M}} = \{\Psi^\mathrm{dec}(z_r)\in\mathbb{R}^{2N}:u_r\in\mathrm{R}^{2n}\}`` where ``z_r`` is the reduced state of the system. By a slight abuse of notation we also denote ``\mathcal{M}`` by ``\mathcal{M}``. 

Similar to the regular PDE case we again build an encoder ``\mathcal{P} \equiv \Psi^\mathrm{enc}`` and a decoder ``\mathcal{R} \equiv \Psi^\mathrm{dec}``; but now both these mappings are required to be symplectic!

Concretely this means: 
1. The encoder is a mapping from a high-dimensional symplectic space to a low-dimensional symplectic space, i.e. ``\Psi^\mathrm{enc}:\mathbb{R}^{2N}\to\mathbb{R}^{2n}`` such that ``\nabla\Psi^\mathrm{enc}\mathbb{J}_{2N}(\nabla\Psi^\mathrm{enc})^T = \mathbb{J}_{2n}``.
2. The decoder is a mapping from a low-dimensional symplectic space to a high-dimensional symplectic space, i.e. ``\Psi^\mathrm{dec}:\mathbb{R}^{2n}\to\mathbb{R}^{2N}`` such that ``(\nabla\Psi^\mathrm{dec})^T\mathbb{J}_{2N}\nabla\Psi^\mathrm{dec} = \mathbb{J}_{2n}``.

If these two maps are constrained to linear maps this amounts to PSD.

## Library Functions

```@docs
GeometricMachineLearning.SymplecticEncoder
GeometricMachineLearning.SymplecticDecoder
GeometricMachineLearning.NonLinearSymplecticEncoder
GeometricMachineLearning.NonLinearSymplecticDecoder
HRedSys
GeometricMachineLearning.build_v_reduced
GeometricMachineLearning.build_f_reduced
PSDLayer
PSDArch
solve!
```

## References 

```@bibliography
Pages = []
Canonical = false

buchfink2023symplectic
peng2016symplectic
```