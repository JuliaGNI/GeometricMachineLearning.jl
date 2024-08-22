# Symplectic Systems

*Symplectic systems* are ODEs whose vector field has a specific structure that is very restrictive. Before we introduce symplectic vector fields on manifolds we first have to define what a *symplectic structure* is:

```@eval
Main.definition(raw"A **symplectic structure** or **symplectic 2-form** ``\Omega`` assigns to each ``x\in\mathcal{M}`` a mapping 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Omega_x:T_x\mathcal{M}\times{}T_x\mathcal{M} \to \mathbb{R},
" * Main.indentation * raw"```
" * Main.indentation * raw"such that ``\Omega_x`` is skew-symmetric and nondegenerate, and ``\Omega`` is closed.")
```

We forego the precise definition of *closedness* because it would require us to introduce differential forms [arnold1978mathematical, bishop1980tensor](@cite). This property is also closely related to the *Jacobi identity* [kraus2017gempic; Chapter 4.4](@cite). After having defined a symplectic structure, we can introduce *Hamiltonian vector fields*[^1]:

[^1]: Also compare this to the definition of the [Riemannian gradient](@ref "The Riemannian Gradient").

```@eval
Main.definition(raw"A **Hamiltonian vector field** at ``x\in\mathcal{M}`` corresponding to the function ``H:\mathcal{M}\to\mathbb{R}`` (called **the Hamiltonian**) is a vector field that has the following property:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Omega(X_H, \dot{\gamma}(0)) = \frac{d}{dt}\bigg|_{t = 0}H(\gamma(t)),
" * Main.indentation * raw"\Omega(X_H, \dot{\gamma}(0)) = \frac{d}{dt}\bigg|_{t = 0}H(\gamma(t)),
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\gamma`` is a ``C^\infty`` curve through ``x``.")
```

Of particular importance for us here are *canonical Hamiltonian systems*:

```@eval
Main.example(raw"To obtain a canonical Hamiltonian system we take ``\mathcal{M} = \mathbb{R}^{2d}`` and 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Omega_z = \mathbb{J}_{2d}^T = \begin{pmatrix} \mathbb{O} & -\mathbb{I} \\ \mathbb{I} & \mathbb{O} \end{pmatrix}
" * Main.indentation * raw"``` 
" * Main.indentation * raw"for all ``z``. We call ``\mathbb{J}_{2d}`` the **Poisson tensor**. In this case the vector field can be written as:
" * Main.indentation * raw"```math
" * Main.indentation * raw"X_H(z) = \mathbb{J}_{2d}\nabla_z{}H,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\nabla{}H`` is the Euclidean gradient.")
```

Typically we further split the ``z`` coordinates on ``\mathbb{R}^{2d}`` into ``(q, p)`` coordinates, where ``q`` and ``p`` are the first ``d`` components of ``z`` and the second ``d`` components of ``z`` respectively, i.e. 

```math
z = \begin{pmatrix} q \\ p \end{pmatrix}.
```

We can then reformulate a Hamiltonian vector field as two separate vector fields:

```math
\begin{aligned}
    \dot{q} & = \frac{\partial{}H}{\partial{}p} \quad\text{and} \\
    \dot{p} & = - \frac{\partial{}H}{\partial{}q}.
\end{aligned}
```

## Solution of Symplectic Systems

The [flow](@ref "The Existence-And-Uniqueness Theorem") of a Hamiltonian ODE has very restrictive properties, the most important one of these is called *symplecticity* [hairer2006geometric](@cite). This property dramatically restricts the dynamically accessible states of the flow map. For a canonical Hamiltonian system symplecticity is defined as follows:


```@eval
Main.definition(raw"A map ``\phi:\mathbb{R}^{2d}\to\mathbb{R}^{2d}`` is called **symplectic** on ``U\subset\mathbb{R}^{2d}`` if
" * Main.indentation * raw"```math
" * Main.indentation * raw"    (\nabla_z\phi)^T\mathbb{J}_{2d}\nabla_z\phi = \mathbb{J}_{2d},
" * Main.indentation * raw"    (\nabla_z\phi)^T\mathbb{J}_{2d}\nabla_z\phi = \mathbb{J}_{2d},
" * Main.indentation * raw"```
" * Main.indentation * raw"for all ``z\in{}U.``")
```

A similar definition of symplecticity holds for the general case of symplectic manifolds ``(\mathcal{M}, \Omega)`` [arnold1978mathematical](@cite). The following holds:

```@eval
Main.theorem(raw"The flow of a Hamiltonian system is symplectic")
```

```@eval
Main.proof(raw"We proof this statement only for canonical Hamiltonian systems here. Consider the flow of a Hamiltonian ODE ``\varphi^t:\mathbb{R}\to\mathbb{R}``. For this we have:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\begin{aligned}
" * Main.indentation * raw"    \frac{d}{dt}\left( (\nabla\varphi^t)^T\mathbb{J}\nabla\varphi^t \right)  & = (\mathbb{J}\nabla^2H)^T\mathbb{J}\nabla\varphi^t + (\nabla\varphi^t)^T\mathbb{J}\mathbb{J}\nabla^2H \\
" * Main.indentation * raw"        & = \nabla^2H\nabla\varphi^t - (\nabla\varphi^t)^T\nabla^2H,
" * Main.indentation * raw"\end{aligned}
" * Main.indentation * raw"```
" * Main.indentation * raw"and this expression is zero at ``t=0.`` This computation holds for all points on the domain on which ``H`` is defined and ``\varphi^t`` is symplectic.")
```

The discipline of finding numerical approximations of flows ``\varphi^t`` such that these numerical approximations also preserve certain properties of that flow (such as symplecticity) is referred to as *structure-preserving numerical integration* or *geometric numerical integration* [hairer2006geometric](@cite). The `Julia` library `GeometricIntegrators` [Kraus:2020:GeometricIntegrators](@cite) offers a wide array of such geometric numerical integrators for a broad class of systems (not just canonical Hamiltonian systems).

It is important to note that symplecticity is a very strong property that may not be achievable in some practical applications. If preservation of symplecticity is not achievable, it may however still be advantageous to consider weaker properties such as [volume preservation](@ref "Divergence-Free Vector Fields").

## References
```@docs
PoissonTensor
GeometricMachineLearning.QPT
GeometricMachineLearning.QPTOAT
```

## Library Functions

```@bibliography
Canonical = false
Pages = []

arnold1978mathematical
bishop1980tensor
hairer2006geometric
```