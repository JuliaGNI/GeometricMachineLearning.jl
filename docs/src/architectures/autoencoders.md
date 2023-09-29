# Variational Autoencoders 

Variational autoencoders (Lee and Carlberg, 2020) train on the following set: 
```math
\mathcal{X}(\mathbb{P}_\mathrm{train}) := \{\mathbf{x}^k(\mu) - \mathbf{x}^0(\mu):0\leq{}k\leq{}K,\mu\in\mathbb{P}_\mathrm{train}\},
```
where $\mathbf{x}^k(\mu)\approx\mathbf{x}(t^k;\mu)$. Note that $\mathbf{0}\in\mathcal{X}(\mathbb{P}_\mathrm{train})$ as $k$ can also be zero. 

The encoder $\Psi^\mathrm{enc}$ and decoder $\Psi^\mathrm{dec}$ are then trained on this set $\mathcal{X}(\mathbb{P}_\mathrm{train})$ by minimizing the reconstruction error: 
```math 
|| \mathbf{x} - \Psi^\mathrm{dec}\circ\Psi^\mathrm{enc}(\mathbf{x}) ||\text{ for $\mathbf{x}\in\mathcal{X}(\mathbb{P}_\mathrm{train})$}.
```

## Initial condition

No matter the parameter $\mu$ the initial condition in the reduced system is always $\mathbf{x}_{r,0}(\mu) = \mathbf{x}_{r,0} = \Psi^\mathrm{enc}(\mathbf{0})$. 

## Reconstructed solution
In order to arrive at the reconstructed solution one first has to **decode** the reduced state and then add the reference state:
```math
\mathbf{x}^\mathrm{reconstr}(t;\mu) = \mathbf{x}^\mathrm{ref}(\mu) + \Psi^\mathrm{dec}(\mathbf{x}_r(t;\mu)),
```
where $\mathbf{x}^\mathrm{ref}(\mu) = \mathbf{x}(t_0;\mu) - \Psi^\mathrm{dec}\circ\Psi^\mathrm{dec}(\mathbf{0})$.

## Symplectic reduced vector field 

A **symplectic vector field** is one whose flow conserves the symplectic structure $\mathbb{J}$. This is equivalent[^1] to there existing a Hamiltonian $H$ s.t. the vector field $X$ can be written as $X = \mathbb{J}\nabla{}H$.
[^1]: Technically speaking the definitions are equivalent only for simply-connected manifolds, so also for vector spaces.   

If the full-order Hamiltonian is $H^\mathrm{full}\equiv{}H$ we can obtain another Hamiltonian on the reduces space by simply setting: 
```math 
H^\mathrm{red}(\mathbf{x}_r(t;\mu)) = H(\mathbf{x}^\mathrm{reconstr}(t;\mu)) = H(\mathbf{x}^\mathrm{ref}(\mu) + \Psi^\mathrm{dec}(\mathbf{x}_r(t;\mu))).
```
The ODE associated to this Hamiltonian is also the one corresponding to **Manifold Galerkin ROM** (see (Lee and Carlberg, 2020)).

## Manifold Galerkin ROM 

Define the **FOM ODE residual** as: 
```math
r: (\mathbf{v}, \xi, \tau; \mu) \mapsto \mathbf{v} - f(\xi, \tau; \mu).
```
The reduced ODE is then defined to be: 
```math
\dot{\hat{\mathbf{x}}}(t;\mu) = \mathrm{arg\,{}min}_{\hat{\mathbf{v}}\in\mathbb{R}^p}|| r(\mathcal{J}(\hat{\mathbf{x}}(t;\mu))\hat{\mathbf{v}},\hat{\mathbf{x}}^\mathrm{ref}(\mu) + \Psi^\mathrm{dec}(\hat{\mathbf{x}}(t;\mu)),t;\mu) ||_2^2,
```
where $\mathcal{J}$ is the Jacobian of the decoder $\Psi^\mathrm{dec}$. This leads to: 
```math 
\mathcal{J}(\hat{\mathbf{x}}(t;\mu))\hat{\mathbf{v}} - f(\hat{\mathbf{x}}^\mathrm{ref}(\mu) + \Psi^\mathrm{dec}(\hat{\mathbf{x}}(t;\mu)), t; \mu) \overset{!}{=} 0 \implies 
\hat{\mathbf{v}} = \mathcal{J}(\hat{\mathbf{x}}(t;\mu))^+f(\hat{\mathbf{x}}^\mathrm{ref}(\mu) + \Psi^\mathrm{dec}(\hat{\mathbf{x}}(t;\mu)), t; \mu),
```
where $\mathcal{J}(\hat{\mathbf{x}}(t;\mu))^+$ is the pseudoinverse of $\mathcal{J}(\hat{\mathbf{x}}(t;\mu))$. Because $\mathcal{J}(\hat{\mathbf{x}}(t;\mu))$ is a symplectic matrix the pseudoinverse is the **symplectic inverse** (see (Peng and Mohseni, 2016)).

Furthermore, because $f$ is Hamiltonian, the vector field describing $dot{\hat{\mathbf{x}}}(t;\mu)$ will also be Hamiltonian. 


## References 
- K. Lee and K. Carlberg. “Model reduction of dynamical systems on nonlinear manifolds using
deep convolutional autoencoders”. In: Journal of Computational Physics 404 (2020), p. 108973.
- Peng L, Mohseni K. Symplectic model reduction of Hamiltonian systems[J]. SIAM Journal on Scientific Computing, 2016, 38(1): A1-A27.