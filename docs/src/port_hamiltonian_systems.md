# Using Symplectic Autoencoders for Port-Hamiltonian Systems

Symplectic autoencoders can also be used to reduce *port-Hamiltonian systems* [van2014port](@cite). Here we focus on *linear port-Hamiltonian systems*[^1]. These are of the form:

[^1]: For a broader class of such systems see [morandin2023modeling](@cite). A generalization to manifolds of such systems is also possible [yoshimura2006diracI, yoshimura2006diracII](@cite).

```math
\Sigma_\mathrm{lpH}(\mathbb{R}^{2N}) = \Sigma_\mathrm{lpH} : \begin{cases} \dot{\hat{z}}(t) & =  (\mathbb{J}_{2N} - \hat{R})\nabla{}H(\hat{z}(t)) + \hat{B}u(t) \\ y(t) & = \hat{B}^T\nabla{}H(\hat{z}(t)),  \end{cases}
```

where ``\mathbb{J}_{2N}`` is the [Poisson tensor](@ref "Symplectic Systems") and ``\hat{R}\in\mathbb{R}^{2N\times{}2N}`` is symmetric semi-positive definite (i.e. all its eigenvalues are non-negative). ``\hat{z}\in\mathbb{R}^{2N}`` is called the *state of the system*, ``u\in\mathbb{R}^m`` are the *system inputs*, ``y\in\mathbb{R}^m`` are the *system outputs*, and ``\hat{B}\in\mathbb{R}^{2N\times{}m}`` connects the inputs to the state. We also refer to *linear port-Hamiltonian systems* as *lpH systems*.

Similar to energy conservation of standard Hamiltonian systems, lpH systems have an associated *energy balance equation*:

```@eval
Main.definition(raw"The **energy balance equation** of a lpH system is:
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

The analogue to the [Poisson tensor](@ref "Symplectic Systems") for lpH systems are so-called *Dirac structures*:

```@eval
llangle = Main.output_type == :latex ? raw"\llangle" : raw"\langle\langle"
rrangle = Main.output_type == :latex ? raw"\rrangle" : raw"\rangle\rangle"
Main.definition(raw"""A Dirac structure for a vector space ``\mathbb{R}^{n}`` is a subspace ``D\subset\mathbb{R}^n\times(\mathbb{R}^n)^* \simeq \mathbb{R}^{2n}`` such that
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""D^\perp = D,
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""i.e. the orthogonal complement of ``D`` is equal to itself. Here the orthogonal complement is taken with respect to the pairing:
""" * Main.indentation * raw"""```math
""" * Main.indentation * llangle * raw"""\cdot,\cdot""" * rrangle * raw""":\mathbb{R}^{2n}\times\mathbb{R}^{2n}\to\mathbb{R}, (e, f)\times(\tilde{e}, \tilde{f}) \mapsto e^T\tilde{f} + \tilde{e}^Tf.
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""Note that ``""" * llangle * raw"""\cdot, \cdot""" * rrangle * raw"""`` is a symmetric bilinear form.""")
```

```@eval
Main.example(raw"""The space:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""    D = \{ (e, f): e\in\mathbb{R}^{2n}, f = \mathbb{J}_{2n}e \}\subset\mathbb{R}^{4n}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""forms a Dirac structure. Note that we also have:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""    (\nabla_z{H}, X_H(z)) = (\nabla_zH, \mathbb{J}\nabla_zH) \in D.
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""So every Hamiltonian System has an associated Dirac structure.""")
```

```@eval
Main.example(raw"""For the lpH shown above we have the relation:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw""" \begin{pmatrix} f \\ y \\e \end{pmatrix} = \begin{pmatrix} \mathbb{J}_{2N}^T & -B & -\mathbb{I}_{2N} \\ B^T & \mathbb{O} & \mathbb{O} \\ \mathbb{I}_{2N} & \mathbb{O} & \mathbb{O} \end{pmatrix} \begin{pmatrix} \bar{e} \\ u \\ \bar{\bar{e}} \end{pmatrix},
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""where we further have the constraints and identifications ``f = -\dot{z},`` ``\bar{e} = \nabla_zH`` and ``\bar{\bar{e}} = Re`` to fully describe the lpH.""")
```

In numerically solving lpH systems the Dirac structure takes a similar role to the symplectic structure of canonical Hamiltonian systems [kotyczka2019discrete](@cite) and the energy-balance equation takes a similar role to energy conservation for canonical Hamiltonian systems. 

Model order reduction of port-Hamiltonian systems can be divided into two approaches: *projection-based methods* and *interpolations of the transfer function* [moser2023structure](@cite). The first approach is equivalent to [Galerkin projection](@ref "Obtaining the Reduced System via Galerkin Projection") and we limit the discussion here to this approach. Similar to the case of [canonical Hamiltonian systems](@ref "Workflow for Symplectic ROM"), we reduce the system with a [symplectic autoencoder](@ref "The Symplectic Autoencoder").

When discussing [symplectic model order reduciton](@ref "Hamiltonian Model Order Reduction") we showed that a Hamiltonian system on the reduced space ``\mathbb{R}^{2n}`` is equivalent to a Hamiltonian system on ``\mathcal{M} = \mathcal{R}(\mathbb{R}^{2n}),`` where ``\mathcal{R}`` is the *reconstruction* in a reduced order modeling framework. Similar statements are also true for lpH systems.

We will now demonstrate how to obtain a reduced-order lpH system from a full-order lpH system and vice-versa:

```@example
Main.include_graphics("tikz/lpH_equivalence"; width = .3, caption = raw"We can derive full lpH systems from reduced lpH systems and vice-versa (in some cases). The solid arrows indicate that we have an explicit construction available, the dashed arrow indicates that in this specific case we do not yet know if a structure-preserving reduction is possible. ") # hide
```

The figure above indicates that we can derive a full system ``\tilde{\Sigma}_\mathrm{lpH}(\mathbb{R}^{2N}) := \Sigma_\mathrm{lpH}(\mathcal{M})`` from a reduced one ``\Sigma_\mathrm{lpH}(\mathbb{R}^{2n}).`` If we have ``R = 0,`` i.e. if the dissipative part of the system is zero, then we can also derive a reduced system ``\Sigma^{R=0}_\mathrm{lpH}(\mathbb{R}^{2n})`` from a full one ``\tilde{\Sigma}^{R=0}_\mathrm{lpH}(\mathbb{R}^{2N}) = \Sigma^{R=0}_\mathrm{lpH}(\mathcal{M}).`` If and when this is true for ``R\neq0`` is an open question[^2]. We now proceed with showing this equivalence, first for the special case ``R = 0.``

[^2]: We indicate this with a dashed arrow.

## The Special Case ``R = 0``

We first focus on the case where ``R = 0.`` This case was also discussed in [kotyczka2019discrete](@cite).

```@eval
Main.theorem(raw"For ``R = 0,`` model reduction of a lpH system with a symplectic autoencoder ``(\Psi^e, \Psi^d)`` yields a lpH system in reduced dimension of the form:
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

Here we show that we can construct a lpH system on the full space from a lpH system on the reduced space; this holds for both ``R=0`` and ``R\neq0.`` The corresponding proof was already introduced in similar form by [rettberg2024data](@cite).

```@eval
Main.theorem(raw"A lpH system on the reduced space induces a lpH system on the full space.")
```

```@eval
Main.proof(raw"""Consider a reduced lpH system:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\Sigma_\mathrm{lpH} : \begin{cases} \dot{z}(t) & =  (\mathbb{J}_{2n} - R)\nabla{}H(z(t)) + Bu(t) \\ y(t) & = B^T\nabla{}H(z(t)),  \end{cases}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""where ``R\in\mathbb{R}^{2n\times2n}`` and ``B\in\mathbb{R}^{2n\times{}m}.`` After multiplying the first equation with ``\nabla_z\mathcal{R}`` from the left we get:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\frac{d}{dt} \mathcal{R}(z(t)) = \nabla_z\mathcal{R}(\mathbb{J}_{2n} - R)\nabla_zH + (\nabla_z\mathcal{R})Bu(t).
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""From now on we call ``\tilde{B} := (\nabla_z\mathcal{R})B.`` We then look at the terms (i) ``(\nabla_z\mathcal{R})\mathbb{J}_{2n}\nabla_zH`` and (ii) ``(\nabla_z\mathcal{R})R\nabla_zH.`` The first one (i) becomes:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\begin{aligned}
""" * Main.indentation * raw"""(\nabla_z\mathcal{R})\mathbb{J}_{2n}\nabla_zH & = \mathbb{J}_{2N}\mathbb{J}_{2N}^T(\nabla_z\mathcal{R})\mathbb{J}_{2n}\nabla_zH \\
""" * Main.indentation * raw"""                                            & = \mathbb{J}_{2N}((\nabla_z\mathcal{R})^+)^T\nabla_zH \\
""" * Main.indentation * raw"""                                            & = \mathbb{J}_{2N}\nabla_{\mathcal{R}(z)}(H\circ\psi)
""" * Main.indentation * raw"""\end{aligned}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""And the second one (ii) becomes:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\begin{aligned}
""" * Main.indentation * raw"""(\nabla_z\mathcal{R})R\nabla_zH   & = (\nabla_z\mathcal{R})R\nabla_z(H\circ\psi\circ\mathcal{R}) \\
""" * Main.indentation * raw"""                                & = (\nabla_z\mathcal{R})R(\nabla_z\mathcal{R})^T\nabla_{\mathcal{R}(z)}(H\circ\psi) \\
""" * Main.indentation * raw"""                                & =: \tilde{R}\nabla_{\mathcal{R}(z)}\bar{H}.
""" * Main.indentation * raw"""\end{aligned}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""We then have in total:
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""\tilde{\Sigma}_\mathrm{lpH}:\begin{cases} \frac{d}{dt}\mathcal{R}(z(t)) & = (\mathbb{J}_{2N} - \tilde{B})\nabla_{\mathcal{R}(z)}\bar{H} + \tilde{B}u(t) \\ \tilde{y}(t) & = \tilde{B}^Tu(t) \end{cases}
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""where  ``\tilde{R}|_{\mathcal{R}(z)} = (\nabla_z\mathcal{R})^TR(\nabla_z\mathcal{R}),`` ``\tilde{B} := (\nabla_z\mathcal{R})B`` and ``\bar{H} = H\circ\psi``.""")
```

As was already discussed in [the section on Hamiltonian model order reduction](@ref "The Symplectic Solution Manifold") the encoder ``\Psi^e`` can be constructed such that it is exactly the local inverse ``\psi.`` This was done in e.g. [otto2023learning](@cite). Enforcing this for symplectic autoencoders is also [straightforward](@ref "The Symplectic Autoencoder").
