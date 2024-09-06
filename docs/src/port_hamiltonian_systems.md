# Using Symplectic Autoencoders for Port-Hamiltonian Systems

Symplectic autoencoders can also be used to reduce *port-Hamiltonian systems* [van2014port](@cite). Here we focus on *linear port-Hamiltonian systems*[^1] which are of the form:

[^1]: For a broader class of such systems see [morandin2023modeling](@cite). A generalization to manifolds of such systems is also possible by utilizing *Dirac structures* [yoshimura2006diracI, yoshimura2006diracII](@cite).

```math
\Sigma_\mathrm{lpH} : \begin{cases} \dot{\hat{z}}(t) & =  (\mathbb{J}_{2N} - \hat{R})\nabla{}H(\hat{z}(t)) + \hat{B}u(t) \\ y(t) & = \hat{B}^T\nabla{}H(\hat{z}(t)),  \end{cases}
```

where ``\mathbb{J}_{2N}`` is the [Poisson tensor](@ref "Symplectic Systems"), ``\hat{R}\in\mathbb{R}^{2N\times{}2N}`` is symmetric semi-positive definite (i.e. all its eigenvalues are non-zero), ``\hat{z}\in\mathbb{R}^{2N}`` is called the *state of the system*, ``u\in\mathbb{R}^m`` are the *system inputs*, ``y\in\mathbb{R}^m`` are the *system outputs*, and ``\hat{B}\in\mathbb{R}^{2N\times{}m}`` connects the inputs to the state. We also refer to *linear port-Hamiltonian systems* as *lpH systems*.

Similar to energy conservation of standard Hamiltonian systems, lpH systems have an associated *energy balance equation*:

```@eval
Main.definition(raw"The energy balance equation of a lpH system is:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\frac{d}{dt}H(z(t)) \leq y(t)^Tu(t).
" * Main.indentation * raw"```
" * Main.indentation * raw"If we further have ``R = 0,`` then the inequality becomes an equality.")
```

```@eval
Main.proof(raw"""We evaluate the derivative of ``H(z(t))`` with respect to ``t``:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\begin{aligned}
""" * Main.indentation * raw"""\frac{d}{dt}H(\varphi^t(z_0))& = \nabla{}H(\varphi^t(z_0))\frac{d}{dt}\varphi^t(z_0) \\
""" * Main.indentation * raw"""                             & = \nabla{}H(z(t))(\mathbb{J} - R)\nabla{}H(z(t)) + (\nabla{}H(z(t)))^TBu(t) \\
""" * Main.indentation * raw"""                             & = (\nabla{}H(z(t)))^TR\nabla{}H(z(t)) + \underbrace{(\nabla{H}(z(t)))^TB}_{y(t)^T}u(t) \\
""" * Main.indentation * raw"""                             & \leq y(t)^Tu(t),
""" * Main.indentation * raw"""\end{aligned}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""where we used that ``R`` is symmetric and positive semi-definite in the last step.""")
```

Model order reduction of port-Hamiltonian systems can be divided into two approaches: *projection-based methods* and *interpolations of the transfer function* [moser2023structure](@cite). The first approach equivalent to [Galerkin projection](@ref "Obtaining the Reduced System via Galerkin Projection") and we limit the discussion here to this approach. Similar to the case of [canonical Hamiltonian systems](@ref "Workflow for Symplectic ROM"), we reduce the system with a [symplectic autoencoder](@ref "The Symplectic Autoencoder"):

## ``R = 0`` as a Special Case

We first focus on the case where ``R = 0.`` This case was also discussed in [kotyczka2019discrete](@cite).

```@eval
Main.theorem(raw"For ``R = 0,`` model reduction of a lpH system with a symplectic autoencoder ``(\Psi^e, \Psi^d)`` yields an lpH system in reduced dimension of the form:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \bar{\Sigma}_\mathrm{lpH} : \begin{cases} \dot{z}(t) & = \mathbb{J}_{2n}\bar{H}(z(t)) + Bu(t) \\ y(t) & = B^T\nabla\bar{H}(z(t)) \end{cases},
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``B := (\nabla\Psi^d)^+\hat{B}`` and ``\bar{H}(z(t)) := H(\Psi^d(z(t))).``") 
```

``(\nabla\Psi^d)^+ = \mathbb{J}_{2n}^T(\nabla\Psi^d)^T\mathbb{J}_{2N}`` is the [symplectic inverse](@ref "The Symplectic Solution Manifold").

```@eval
Main.proof(raw"We have to proof that the dynamics of ``\Psi^d(z)``, that approximate ``\hat{z}``, are described by a lpH system. We first insert ``\bar{z}(t) \approx \Psi^d(z(t))`` into the first equation of ``\Sigma_{lpH}``:
" * Main.indentation * raw"```math
" * Main.indentation * raw"(\nabla\Psi^d)\dot{z}(t) = \mathbb{J}_{2N}\nabla{}H(\Psi^d(z(t))) + Bu(t).
" * Main.indentation * raw"```
" * Main.indentation * raw"We then multiply the equation above with the *symplectic inverse* ``(\nabla\Psi^d)^+``:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\begin{aligned}
" * Main.indentation * raw"    \dot{z}(t)  & = (\nabla\Psi^d)^+\mathbb{J}_{2N}\nabla{}H(\Psi^d(z(t))) + (\nabla\Psi^d)^+\hat{B}u(t) \\
" * Main.indentation * raw"                & = \mathbb{J}_{2n}(\nabla\Psi^d)^T\nabla{}H(\Psi^d(z(t))) + (\nabla\Psi^d)^+\hat{B}u(t) \\
" * Main.indentation * raw"                & = \mathbb{J}_{2n}\nabla(H\circ\Psi^d(z(t))) + Bu(t),
" * Main.indentation * raw"\end{aligned}
" * Main.indentation * raw"```
" * Main.indentation * raw"thus proving our assertion.")
```

## From the Reduced Space to the Full Space

Here we recall a proof from [rettberg2024data](@cite).

```@eval
Main.theorem(raw"An lpH system on the reduced space induces an lpH system on the full space.")
```

```@eval
Main.proof(raw"""Consider a reduced lpH system:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\Sigma_\mathrm{lpH} : \begin{cases} \dot{z}(t) & =  (\mathbb{J}_{2n} - R)\nabla{}H(z(t)) + Bu(t) \\ y(t) & = B^T\nabla{}H(z(t)),  \end{cases}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""where ``R\in\mathbb{R}^{2n\times2n}`` and ``B\in\mathbb{R}^{2n\times{}m}.`` After multiplying the first equation with ``\nabla_z\mathcal{R}`` from the left we get:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\begin{aligned}
""" * Main.indentation * raw"""\frac{d}{dt} \mathcal{R}(z(t))  & = \nabla_z\mathcal{R}(\mathbb{J}_{2n} - R)\nabla{}H(z(t)) + \nabla_z\mathcal{R}Bu(t) \\
""" * Main.indentation * raw"""                                & = \nabla_z\mathcal{R}(\mathbb{J}_{2n} - R)(\nabla_z\mathcal{R})^T(\nabla_{\mathcal{R}(z)}\psi)^T\nabla{}H(\psi(\mathcal{R}(z(t))))  \\
""" * Main.indentation * raw"""                                & = \nabla_z\mathcal{R}(\mathbb{J}_{2n} - R)(\nabla_z\mathcal{R})^T\nabla_{\mathcal{R}(z)}(H\circ\psi) \\
""" * Main.indentation * raw"""                                & = (\mathcal{J}_{2N}|_{\mathcal{M}} - \tilde{R})nabla_{\mathcal{R}(z)}\bar{H},
""" * Main.indentation * raw"""\end{aligned}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""where ``\tilde{R}|_{\mathcal{R}(z)} = ((\nabla\mathcal{R})^+)^TR(\nabla\mathcal{R})^+`` and ``\bar{H} = H\circ\psi``.""")
```

As was already discussed in [the section on Hamiltonian model order reduction](@ref "The Symplectic Solution Manifold") the encoder ``\Psi^e`` can be constructed such that it is exactly the local inverse ``\varphi.`` This was done in e.g. [otto2023learning](@cite). Enforcing this for [symplectic autoencoders](@ref "The Symplectic Autoencoder") is also straightforward:
