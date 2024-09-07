# Using Symplectic Autoencoders for Port-Hamiltonian Systems

Symplectic autoencoders can also be used to reduce *port-Hamiltonian systems* [van2014port](@cite). Here we focus on *linear port-Hamiltonian systems*[^1] which are of the form:

[^1]: For a broader class of such systems see [morandin2023modeling](@cite). A generalization to manifolds of such systems is also possible by utilizing *Dirac structures* [yoshimura2006diracI, yoshimura2006diracII](@cite).

```math
\Sigma_\mathrm{lpH}(\mathbb{R}^{2N}) = \Sigma_\mathrm{lpH} : \begin{cases} \dot{\hat{z}}(t) & =  (\mathbb{J}_{2N} - \hat{R})\nabla{}H(\hat{z}(t)) + \hat{B}u(t) \\ y(t) & = \hat{B}^T\nabla{}H(\hat{z}(t)),  \end{cases}
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

Model order reduction of port-Hamiltonian systems can be divided into two approaches: *projection-based methods* and *interpolations of the transfer function* [moser2023structure](@cite). The first approach equivalent to [Galerkin projection](@ref "Obtaining the Reduced System via Galerkin Projection") and we limit the discussion here to this approach. Similar to the case of [canonical Hamiltonian systems](@ref "Workflow for Symplectic ROM"), we reduce the system with a [symplectic autoencoder](@ref "The Symplectic Autoencoder").

When discussing [symplectic model order reduciton](@ref "Hamiltonian Model Order Reduction") we showed that a Hamiltonian system on the reduced space ``\mathbb{R}^{2n}`` is equivalent to a Hamiltonian system on ``\mathcal{M} = \mathcal{R}(\mathbb{R}^{2n}).``

We will now show the following equivalence relationships:

```@example
Main.include_graphics("tikz/lpH_equivalence"; width = .3, caption = raw"We can derive full lpH systems from reduced lpH systems and vice-versa (in some cases). ") # hide
```

The figure above indicates that we can derive a full system ``\tilde{\Sigma}_\mathrm{lpH}(\mathbb{R}^{2N}) := \Sigma_\mathrm{lpH}(\mathcal{M})`` from a reduced one ``\Sigma_\mathrm{lpH}(\mathbb{R}^{2n}).`` If we have ``R = 0,`` i.e. if the dissipative part of the system is zero, then we can also derive a reduced system ``\Sigma^{R=0}_\mathrm{lpH}(\mathbb{R}^{2n})`` from a full one ``\tilde{\Sigma}^{R=0}_\mathrm{lpH}(\mathbb{R}^{2N}) = \Sigma^{R=0}_\mathrm{lpH}(\mathcal{M}).`` When this is true for ``R\neq0`` is an open question. We now proceed with showing this equivalence, first for the special case ``R = 0.``

## The Special Case ``R = 0``

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

## From the Reduced Space to the Full Space for ``R\neq0``

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
""" * Main.indentation * raw"""\frac{d}{dt} \mathcal{R}(z(t)) = \nabla_z\mathcal{R}(\mathbb{J}_{2n} - R)\nabla_zH + \nabla_z\mathcal{R}Bu(t).
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""From now on we call ``\tilde{B} := \nabla_z\mathcal{R}B.`` We then look at the terms (i) ``(\nabla_z\mathcal{R})\mathbb{J}_{2n}\nabla_zH`` and (ii) ``(\nabla_z\mathcal{R})R\nabla_zH.`` The first one (i) becomes:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\begin{aligned}
""" * Main.indentation * raw"""\nabla_z\mathcal{R}\mathbb{J}_{2n}\nabla_zH & = \mathbb{J}_{2N}\mathbb{J}_{2N}^T\mathcal{R}\mathbb{J}_{2n}\nabla_zH \\
""" * Main.indentation * raw"""                                            & = \mathbb{J}_{2N}(\mathcal{R}^+)^T\nabla_zH \\
""" * Main.indentation * raw"""                                            & = \mathbb{J}_{2N}\nabla_{\mathcal{R}(z)}(H\circ\psi)
""" * Main.indentation * raw"""\end{aligned}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""And the second one (ii) becomes:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\begin{aligned}
""" * Main.indentation * raw"""\nabla_z\mathcal{R}R\nabla_zH   & = \nabla_z\mathcal{R}R\nabla_z(H\circ\psi\circ\mathcal{R}) \\
""" * Main.indentation * raw"""                                & = \nabla_z\mathcal{R}R(\nabla_z\mathcal{R})^T\nabla_{\mathcal{R}(z)}(H\circ\psi) \\
""" * Main.indentation * raw"""                                & =: \tilde{R}\nabla_{\mathcal{R}(z)}\bar{H}.
""" * Main.indentation * raw"""\end{aligned}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""We then have in total:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\tilde{\Sigma}_\mathrm{lpH}:\begin{cases} \frac{d}{dt}\mathcal{R}(z(t)) & = (\mathbb{J}_{2N} - \tilde{B})\nabla_{\mathcal{R}(z)}\bar{H} + \tilde{B}u(t) \\ \tilde{y}(t) & = \tilde{B}^Tu(t) \end{cases}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""where  ``\tilde{R}|_{\mathcal{R}(z)} = (\nabla_z\mathcal{R})^TR(\nabla_z\mathcal{R}),`` ``\tilde{B} := (\nabla_z\mathcal{R})B`` and ``\bar{H} = H\circ\psi``.""")
```

As was already discussed in [the section on Hamiltonian model order reduction](@ref "The Symplectic Solution Manifold") the encoder ``\Psi^e`` can be constructed such that it is exactly the local inverse ``\varphi.`` This was done in e.g. [otto2023learning](@cite). Enforcing this for [symplectic autoencoders](@ref "The Symplectic Autoencoder") is also straightforward:
