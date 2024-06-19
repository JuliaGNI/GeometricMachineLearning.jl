# Reduced Order Modeling

Reduced order modeling is a data-driven technique that exploits the structure of parametric partial differential equations (PPDEs) to make repeated simulations of this PPDE much cheaper.

For this consider a PPDE written in the form: ``F(z(\mu);\mu)=0`` where ``z(\mu)`` evolves on a infinite-dimensional Hilbert space ``V``. 

In modeling any PDE we have to choose a discretization (particle discretization, finite element method, ...) of ``V``, which will be denoted by ``V_h``. The space ``V_h`` is not infinite-dimensional but still very large. Solving a discretized PDE in this space is typically very expensive! In reduced order modeling we utilize the fact that slightly different choices of parameters ``\mu`` will give qualitatively similar solutions. We can therefore perform a few simulations in the full space ``V_h`` and then make successive simulations cheaper by *learning* from the past simulations. A crucial concept in this is the *solution manifold*.

## Solution manifold 

To any PPDE and a certain parameter set ``\mathbb{P}`` we associate a *solution manifold*: 

```math 
\mathcal{M} = \{z(\mu):F(z(\mu);\mu)=0, \mu\in\mathbb{P}\}.
```

A motivation for reduced order modeling is that even though the space ``V_h`` is of very high-dimension, the solution manifold will typically be a very small space. The image below shows a two-dimension solution manifold[^1] embedded in ``V_h\equiv\mathbb{R}^3``:

[^1]: The systems be deal with usually have much greater dimension of course. The dimension of ``V_h`` will be in the thousands and the dimension of the solution manifold will be a few order of magnitudes smaller. Because this cannot be easily visualized we resort to showing a two-dimensional manifold in a three-dimensional space here. 

![](../tikz/solution_manifold_2.png)

As an example of this consider the one-dimensional wave equation [blickhan2023registration](@cite): 

```math
\partial_{tt}^2q(t,\xi;\mu) = \mu^2\partial_{\xi\xi}^2q(t,\xi;\mu)\text{ on }I\times\Omega,
```
where ``I = (0,1)$ and $\Omega=(-1/2,1/2)``. As initial condition for the first derivative we have ``\partial_tq(0,\xi;\mu) = -\mu\partial_\xi{}q_0(\xi;\mu)`` and furthermore ``q(t,\xi;\mu)=0`` on the boundary (i.e. ``\xi\in\{-1/2,1/2\}``).

The solution manifold is a 1-dimensional submanifold of an infinite-dimensional function space: 

```math
\mathcal{M} = \{(t, \xi)\mapsto{}q(t,\xi;\mu)=q_0(\xi-\mu{}t;\mu):\mu\in\mathbb{P}\subset\mathbb{R}\}.
```

If we provide an initial condition $u_0$, a parameter instance ``\mu`` and a time ``t``, then ``\xi\mapsto{}q(t,\xi;\mu)`` will be the momentary solution. If we consider the time evolution of ``q(t,\xi;\mu)``, then it evolves on a two-dimensional submanifold ``\bar{\mathcal{M}} := \{\xi\mapsto{}q(t,\xi;\mu):t\in{}I,\mu\in\mathbb{P}\}``.

In reduced order modeling we try to find an approximation to this solution manifolds. Neural networks offer a way of doing so efficiently!

## General workflow

In reduced order modeling we aim to construct an approximation to the solution manifold and that is ideally of a dimension not much greater than that of the solution manifold and the solved so-called *reduced equations* in the small space. This approximation to the solution manifold is performed in the following steps: 
1. Discretize the PDE.
2. Solve the discretized PDE for a certain set of parameter instances ``\mu\in\mathbb{P}``.
3. Build a reduced basis with the data obtained from having solved the discretized PDE. This step consists of finding two mappings: the *reduction* ``\mathcal{P}`` and the *reconstruction* ``\mathcal{R}``.

The third step can be done with various machine learning (ML) techniques. Traditionally the most popular of these has been *Proper orthogonal decomposition* (POD), but in recent years *autoencoders* have also become a popular alternative [fresca2021comprehensive](@cite).

After having obtained ``\mathcal{P}`` and ``\mathcal{R}`` we still need to solve the *reduced system*. Solving the reduced system is typically referred to as the *online phase* in reduced order modeling. This is sketched below: 

```@example
include_graphics(../tikz/offline_online) # hide
```

The online phase is applying the mapping ``\mathcal{NN}`` in the low-dimensional space in order to predict the next time step. Crucially this step can be made very cheap when compared to the full-order model.

## References 
```@bibliography
Pages = []
Canonical = false

fresca2021comprehensive
```