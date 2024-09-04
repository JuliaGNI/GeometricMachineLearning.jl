# Using Symplectic Autoencoders for Port-Hamiltonian Systems

Symplectic autoencoders can also be used to reduce *port-Hamiltonian systems* [van2014port](@cite). Here we focus on *linear port-Hamiltonian systems*[^1] which are of the form:

[^1]: For a broader class of such systems see [morandin2023modeling](@cite). A generalization to manifolds of such systems is also possible by utilizing *Dirac structures* [yoshimura2006diracI, yoshimura2006diracII](@cite).

```math
\Sigma_\mathrm{lpH} : \begin{cases} \dot{\hat{z}}(t) & =  (\mathbb{J}_{2N} - \hat{R})\nabla{}H(\hat{z}(t)) + \hat{B}u(t) \\ y(t) & = \hat{B}^T\nabla{}H(\hat{z}(t)),  \end{cases}
```

where ``\mathbb{J}_{2N}`` is the [Poisson tensor](@ref "Symplectic Systems"), ``\hat{R}\in\mathbb{R}^{2N\times{}2N}`` is symmetric semi-positive definite (i.e. all its eigenvalues are non-zero), ``\hat{z}\in\mathbb{R}^{2N}`` is called the *state of the system*, ``u\in\mathbb{R}^m`` are the *system inputs*, ``y\in\mathbb{R}^m`` are the *system outputs*, and ``\hat{B}\in\mathbb{R}^{2N\times{}m}`` connects the inputs to the state. We also refer to *linear port-Hamiltonian systems* as *lpH systems*.

Model order reduction of port-Hamiltonian systems can be divided into two approaches: *projection-based methods* and *interpolations of the transfer function* [moser2023structure](@cite). The first approach equivalent to [Galerkin projection](@ref "Obtaining the Reduced System via Galerkin Projection") and we limit the discussion here to this approach. Similar to the case of [canonical Hamiltonian systems](@ref "Workflow for Symplectic ROM"), we reduce the system with a [symplectic autoencoder](@ref "The Symplectic Autoencoder"):

```@eval
Main.theorem(raw"Model reduction of a lpH system with a symplectic autoencoder ``(\Psi^e, \Psi^d)`` yields an lpH system in reduced dimension of the form:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \bar{\Sigma}_\mathrm{lpH} : \begin{cases} \dot{z}(t) & = (\mathbb{J}_{2n} - R)\bar{H}(z(t)) + Bu(t) \\ y(t) & = B^T\nabla\bar{\H}(z(t)) \end{cases},
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``R := (\nabla\Psi^d)^+\hat{R}((\nabla\Psi^d)^+)^T,`` ``B := (\nabla\Psi^d)^+\hat{B}`` and ``\bar{H}(z(t)) := H(\Psi^d(z(t))).``") 
```

``(\nabla\Psi^d)^+ = \mathbb{J}_{2n}^T(\nabla\Psi^d)^T\mathbb{J}_{2N}`` is the [symplectic inverse](@ref).

```@eval
Main.proof()
We have to proof that the dynamics of ``\Psi^d(z)``, that approximate ``\hat{z}``, are described by a lpH system. We first insert ``\bar{z}(t) \approx \Psi^d(z(t))`` into the first equation of ``\Sigma_{lpH}``:
```math
(\nabla\Psi^d)\dot{z}(t) = (\mathbb{J}_{2N} - \hat{R})\nabla{}H(\Psi^d(z(t))) + Bu(t).
```
We then multiply the equation above with the *symplectic inverse* ``(\nabla\Psi^d)^+``:
```math
\begin{aligned}
    \dot{z}(t) & = ((\nabla\Psi^d)^+\mathbb{J}_{2N} - (\nabla\Psi^d)^+\hat{R})\nabla{}H(\Psi^d(z(t))) + (\nabla\Psi^d)^+Bu(t) = 
\end{aligned}
```
```