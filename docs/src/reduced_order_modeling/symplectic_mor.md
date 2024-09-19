# Hamiltonian Model Order Reduction 

Hamiltonian PDEs are partial differential equations that, like its ODE counterpart, have a Hamiltonian associated with it. The linear wave equation can be written as such a Hamiltonian PDE with 

```math
\mathcal{H}(q, p; \mu) := \frac{1}{2}\int_\Omega\mu^2(\partial_\xi{}q(t,\xi;\mu))^2 + p(t,\xi;\mu)^2d\xi.
```

Note that in contrast to the ODE case where the Hamiltonian is a function ``H(\cdot; \mu):\mathbb{R}^{2d}\to\mathbb{R},`` we now have a functional ``\mathcal{H}(\cdot, \cdot; \mu):\mathcal{C}^\infty(\mathcal{D})\times\mathcal{C}^\infty(\mathcal{D})\to\mathbb{R}.`` The PDE for this Hamiltonian can be obtained similarly as in the ODE case:

```math
\partial_t{}q(t,\xi;\mu) = \frac{\delta{}\mathcal{H}}{\delta{}p} = p(t,\xi;\mu), \quad \partial_t{}p(t,\xi;\mu) = -\frac{\delta{}\mathcal{H}}{\delta{}q} = \mu^2\partial_{\xi{}\xi}q(t,\xi;\mu)
```

Neglecting the Hamiltonian structure of a system can have grave consequences on the performance of the reduced order model [peng2016symplectic, buchfink2023symplectic, tyranowski2023symplectic](@cite) which is why all algorithms in `GeometricMachineLearning` designed for producing reduced order models respect the structure of the system.

## The Symplectic Solution Manifold 

As with regular parametric PDEs, we also associate a solution manifold with Hamiltonian PDEs. This is a finite-dimensional manifold, on which the dynamics can be described through a Hamiltonian ODE. The reduced system, with which we approximate this symplectic solution manifold, is a low dimensional symplectic vector space ``\mathbb{R}^{2n}`` together with a reduction ``\mathcal{P}`` and a reconstruction ``\mathcal{R}.`` If we now take an initial condition on the solution manifold ``\hat{u}_0\in\mathcal{M} \approx \mathcal{R}(\mathbb{R}^{2n})`` and project it to the reduced space with ``\mathcal{P}``, we get ``u = \mathcal{P}(\hat{u}_0).`` We can now integrate it on the reduced space via the induced differential equation, which is of canonical Hamiltonian form, and obtain an orbit ``u(t)`` which can then be mapped back to an orbit on the solution manifold[^1] via ``\mathcal{R}.`` The resulting orbit ``\mathcal{R}(u(t))`` is ideally the unique orbit on the full order model ``\hat{u}(t)\in\mathcal{M}``.

[^1]: To be precise, an *approximation of the solution manifold* ``\mathcal{R}(\mathbb{R}^{2n})``, as we are not able to find the solution manifold exactly in practice. 


For Hamiltonian model order reduction we additionally require that the reduction ``\mathcal{P}`` satisfies

```math
    \nabla_z\mathcal{P}\mathbb{J}_{2N}(\nabla_z\mathcal{P})^T = \mathbb{J}_{2n} \text{ for $z\in\mathbb{R}^{2N}$}
```

and the reconstruction ``\mathcal{R}`` satisfies[^2]

[^2]: We should note that satisfying this *symplecticity condition* is much more important for the reconstruction than for the reduction. There is a lack of research on whether the symplecticity condition for the projection is really needed; in [buchfink2023symplectic](@cite) it is entirely ignored for example.

```math
    (\nabla_z\mathcal{R})^T\mathbb{J}_{2N}\nabla_z\mathcal{R} = \mathbb{J}_{2n}.
```

With this we have

```@eval
Main.theorem(raw"A Hamiltonian system on the reduced space ``(\mathbb{R}^{2n}, \mathbb{J}_{2n}^T)`` is equivalent to a *non-canonical symplectic system* ``(\mathcal{M}, \mathbb{J}_{2N}^T|_\mathcal{M})`` where 
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathcal{M} = \mathcal{R}(\mathbb{R}^{2n})
" * Main.indentation * raw"```
" * Main.indentation * raw"is an approximation to the solution manifold. We further have
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \mathbb{J}_{2N}|_\mathcal{M}(z) = ((\nabla_z\mathcal{R})^+)^T\mathbb{J}_{2n}(\nabla_z\mathcal{R})^+,
" * Main.indentation * raw"```
" * Main.indentation * raw"so the dynamics on ``\mathcal{M}`` can be described through a Hamiltonian ODE on ``\mathbb{R}^{2n}.``")
```

For the proof we use the fact that ``\mathcal{M} = \mathcal{R}(\mathbb{R}^{2n})`` is a manifold [whose coordinate chart is the local inverse](@ref "The Immersion Theorem") of ``\mathcal{R}`` which we will call ``\psi``, i.e. around a point ``y\in\mathcal{M}`` we have ``\psi\circ\mathcal{R}(y) = y.``[^3] We further define the *symplectic inverse* of a matrix ``A\in\mathbb{R}^{2N\times2n}`` as 

[^3]: A similar proof can be found in [yildiz2024data](@cite). Further note that, if we enforced the condition ``\mathcal{P}\circ\mathcal{R} = \mathrm{id}`` exactly, the projection ``\mathcal{P}`` would be equal to the local inverse ``\psi.`` For the proof here we however only require the existence of ``\psi``, not its explicit construction as ``\mathcal{P}.``

```math
    A^+ = \mathbb{J}_{2n}^TA^T\mathbb{J}_{2N},
```

which gives:

```math
    A^+A = \mathbb{J}_{2n}^TA^T\mathbb{J}_{2N}A = \mathbb{I}_{2n},
```

iff ``A`` is symplectic, i.e. ``A^T\mathbb{J}_{2N}A = \mathbb{J}_{2n}``.

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
" * Main.indentation * raw"which proves that every Hamiltonian system on ``\mathbb{R}^{2n}`` induces a Hamiltonian system on ``\mathcal{M}``. Conversely assume we are given a Hamiltonian vector field whose flow map evolves on ``\mathcal{M}``, which we denote by
" * Main.indentation * raw"```math
" * Main.indentation * raw"\hat{X}(z) = \mathbb{J}_{2N}\nabla_{z}\hat{H} = (\nabla_{\psi(z)}\mathcal{R})\bar{X}(\psi(z)),
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\bar{X}`` is a vector field on the reduced space. In the last equality we used that the flow map evolves on ``\mathcal{M}``, so the corresponding vector field needs to map to ``T\mathcal{M}.`` We further have:
" * Main.indentation * raw"```math
" * Main.indentation * raw"(\nabla_{\psi(z)}\mathcal{R})^+\hat{X}(z) = \mathbb{J}_{2n}(\nabla_{\psi(z)}\mathcal{R})^T\nabla_z\hat{H} = \mathbb{J}_{2n}\nabla_{\psi(z)}(\hat{H}\circ\mathcal{R}) = \bar{X}(\psi(z)),
" * Main.indentation * raw"```
" * Main.indentation * raw"and we see that the vector field on the reduced space also has to be Hamiltonian. We can thus express a high-dimensional Hamiltonian system on ``\mathcal{M}`` with a low-dimensional Hamiltonian system on ``\mathbb{R}^{2n}``.")
```

In the proof we used that *the pseudoinverse is unique*. This is not true in general [peng2016symplectic](@cite), but holds for the architectures discussed here (proper symplectic decomposition and symplectic autoencoders). We will postpone the proof of this until after we introduced [symplectic autoencoders in detail](@ref "The Symplectic Autoencoder").

This theorem serves as the basis for Hamiltonian model order reduction via proper symplectic decomposition and symplectic autoencoders. We will now briefly introduce these two approaches[^4].

[^4]: We will discuss symplectic autoencoders later in a [dedicated section](@ref "The Symplectic Autoencoder").

## Proper Symplectic Decomposition

For proper symplectic decomposition (PSD) the reduction ``\mathcal{P}`` and the reconstruction ``\mathcal{R}`` are constrained to be linear, orthonormal and symplectic. Note that these first two properties are shared with [POD](@ref "Proper Orthogonal Decomposition"). The easiest way[^5] to enforce this is through the so-called "cotangent lift" [peng2016symplectic](@cite): 

[^5]: The original PSD paper [peng2016symplectic](@cite) proposes another approach to build linear reductions and reconstructions with the so-called "complex SVD." In practice this only brings minor advantages over the cotangent lift however [tyranowski2023symplectic](@cite).

```math
\mathcal{R} \equiv \Psi_\mathrm{CL} = \begin{bmatrix} \Phi & \mathbb{O} \\ \mathbb{O} & \Phi \end{bmatrix} \text{ where $\Phi\in{}St(n,N)\subset\mathbb{R}^{N\times{}n}$},
```
i.e. both ``\Phi`` and ``\Psi_\mathrm{CL}`` are elements of the [Stiefel manifold](@ref "The Stiefel Manifold") and we furthermore have ``\Psi_\mathrm{CL}^T\mathbb{J}_{2N}\Psi_\mathrm{CL} = \mathbb{J}_{2n}``, i.e. ``\Psi_\mathrm{CL}`` is symplectic. If the [snapshot matrix](@ref "Snapshot Matrix") is of the form: 

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

with ``\mathtt{nts} := f - 1`` is *the number of time steps*. Then ``\Phi_\mathrm{CL}`` can be computed in a very straight-forward manner:

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

Symplectic Autoencoders are a type of neural network suitable for treating Hamiltonian parametrized PDEs with slowly decaying Kolmogorov ``n``-width. It is based on PSD and [symplectic neural networks](@ref "SympNet Architecture")[^6].

[^6]: We call these SympNets most of the time. This term was coined in [jin2020sympnets](@cite).

Symplectic autoencoders are motivated similarly to standard autoencoders for model order reduction: PSD suffers from similar shortcomings as regular POD. PSD is a linear map and the approximation space ``\tilde{\mathcal{M}}= \{\Psi^\mathrm{dec}(z_r)\in\mathbb{R}^{2N}:z_r\in\mathbb{R}^{2n}\}`` is therefore also linear. For problems with slowly-decaying Kolmogorov ``n``-width this leads to very poor approximations.  

In order to overcome this difficulty we use neural networks, more specifically [SympNets](@ref "SympNet Architecture"), together with cotangent lift-like matrices. The resulting architecture, symplectic autoencoders, are discussed in the [dedicated section on neural network architectures](@ref "The Symplectic Autoencoder").


## Workflow for Symplectic ROM

As with any other data-driven [reduced order modeling technique](@ref "General Workflow") we first discretize the PDE. This should be done with a structure-preserving scheme, thus yielding a (high-dimensional) Hamiltonian ODE as a result. Going back to the example of the linear wave equation, we can discretize this equation with finite differences to obtain a Hamiltonian ODE: 

```math
\mathcal{H}_\mathrm{discr}(z(t;\mu);\mu) := \frac{1}{2}x(t;\mu)^T\begin{bmatrix}  -\mu^2D_{\xi{}\xi} & \mathbb{O} \\ \mathbb{O} & \mathbb{I}  \end{bmatrix} x(t;\mu).
```

In Hamiltonian reduced order modeling we try to find a symplectic submanifold in the solution space[^7] that captures the dynamics of the full system as well as possible.

[^7]: The submanifold, that approximates the solution manifold, is ``\tilde{\mathcal{M}} = \{\Psi^\mathrm{dec}(z_r)\in\mathbb{R}^{2N}:u_r\in\mathrm{R}^{2n}\}`` where ``z_r`` is the reduced state of the system. By a slight abuse of notation we also denote ``\tilde{\mathcal{M}}`` by ``\mathcal{M}`` as we have done previously when showing equivalence between Hamiltonian vector fields on ``\mathbb{R}^{2n}`` and ``\mathcal{M}``. 

Similar to the regular PDE case we again build an encoder ``\mathcal{P} \equiv \Psi^\mathrm{enc}`` and a decoder ``\mathcal{R} \equiv \Psi^\mathrm{dec}``; but now both these mappings are required to be symplectic.

Concretely this means: 
1. The encoder is a mapping from a high-dimensional symplectic space to a low-dimensional symplectic space, i.e. ``\Psi^\mathrm{enc}:\mathbb{R}^{2N}\to\mathbb{R}^{2n}`` such that ``\nabla\Psi^\mathrm{enc}\mathbb{J}_{2N}(\nabla\Psi^\mathrm{enc})^T = \mathbb{J}_{2n}``.
2. The decoder is a mapping from a low-dimensional symplectic space to a high-dimensional symplectic space, i.e. ``\Psi^\mathrm{dec}:\mathbb{R}^{2n}\to\mathbb{R}^{2N}`` such that ``(\nabla\Psi^\mathrm{dec})^T\mathbb{J}_{2N}\nabla\Psi^\mathrm{dec} = \mathbb{J}_{2n}``.

If these two maps are constrained to linear maps this amounts to PSD.

After we obtained ``\Psi^\mathrm{enc}`` and ``\Psi^\mathrm{dec}`` we can construct the reduced model. `GeometricMachineLearning` has a *symplectic Galerkin projection* implemented. This symplectic Galerkin projection does:

```math
    \begin{pmatrix} v_r(q, p) \\ f_r(q, p) \end{pmatrix} = \frac{d}{dt} \begin{pmatrix} q \\ p \end{pmatrix} = \mathbb{J}_{2n}(\nabla_z\mathcal{R})^T\mathbb{J}_{2N}^T \begin{pmatrix} v(\mathcal{R}(q, p)) \\ f(\mathcal{R}(q, p)) \end{pmatrix},
```
where ``v`` are the first ``n`` components of the vector field and ``f`` are the second ``n`` components of the vector field. The superscript ``r`` indicated a *reduced* vector field. These reduced vector fields are built with [`GeometricMachineLearning.build_v_reduced`](@ref) and [`GeometricMachineLearning.build_f_reduced`](@ref).


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

```@raw latex
\section*{Chapter Summary}

In this chapter we introduced the concept of \textit{data-driven reduced order modeling}. It was shown how data-driven reduced order modeling can at least partially be motivated by the concept of a \textit{solution manifold}. \textit{Proper orthogonal decomposition} (POD) and \textit{autoencoders} were discussed as specific examples of performing the offline phase in reduced order modeling. POD is an application of \textit{singular value decomposition} and as such belongs to the realm of linear algebra. Autoencoders are more general approximators that are build with neural networks and have to optimized during a \textit{training stage}.

We furthermore discussed how a reduced order model can be made structure-preserving and discussed \textit{proper symplectic decomposition} (PSD) as the structure-preserving alternative to POD. PSD is a more restrictive form of POD where we impose additional conditions on the reduction $\mathcal{P}$ and reconstruction $\mathcal{R}$ so that a Hamiltonian full order model can be reduced to a Hamiltonian reduced order model. At the very end of the chapter we teased \textit{symplectic autoencoders} which will be discussed in detail in part III. These offer, similar to standard autoencoders, a way of constructing more general Hamiltonian reductions.

\begin{comment}
```

## References 

```@bibliography
Pages = []
Canonical = false

buchfink2023symplectic
peng2016symplectic
```

```@raw latex
\end{comment}
```

```@raw html
<!--
```

# References

```@bibliography
Pages = []
Canonical = false

lee2020model
fresca2021comprehensive
blickhan2023registration
challerjee2000introduction
peng2016symplectic
buchfink2023symplectic
```

```@raw html
-->
```