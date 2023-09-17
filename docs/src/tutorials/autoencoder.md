# Reduced Order Modelling and Autoencoders 

Reduced order modelling is a data-driven technique that exploits the structure of parametric PDEs to make solving those PDEs easier.

Consider a parametric PDE written in the form: $F(z(\mu);\mu)=0$ where $z(\mu)$ evolves on a infinite-dimensional Hilbert space $V$. 

In modelling any PDE we have to choose a discretization (particle discretization, finite element method, ...) of V, which will be denoted by $V_h$. 

## Solution manifold 

To any parametric PDE we associate a **solution manifold**: 

```math 
\mathcal{M} = \{z(\mu):F(z(\mu);\mu)=0, \mu\in\mathbb{P}\}.
```

![](images/solution_manifold_2.png)

In the image above a 2-dimensional solution manifold is visualized as a sub-manifold in 3-dimensional space. In general the embedding space is an infinite-dimensional function space.

As an example of this consider the 1-dimensional wave equation: 

```math
\partial_{tt}^2z(t,\xi;\mu) = \mu^2\partial_{\xi\xi}^2z(t,\xi;\mu)\text{ on }I\times\Omega,
```
where $I = (0,1)$ and $\Omega=(-1/2,1/2)$. As initial condition for the first derivative we have $\partial_tz(0,\xi;\mu) = -\mu\partial_\xi{}z_0(\xi;\mu)$ and furthermore $z(t,\xi;\mu)=0$ on the boundary (i.e. $\xi\in\{-1/2,1/2\}$).

The solution manifold is a 1-dimensional submanifold: 

```math
\mathcal{M} = \{(t, \xi)\mapsto{}u(t,\xi;\mu)=u_0(\xi-\mu{}t;\mu):\mu\in\mathbb{P}\subset\mathbb{R}\}.
```

If we provide an initial condition $u_0$, a parameter instance $\mu$ and a time $t$, then $\xi\mapsto{}z(t,\xi;\mu)$ will be the momentary solution. If we consider the time evolution of $z(t,\xi;\mu)$, then it evolves on a two-dimensional submanifold $\bar{\mathcal{M}} := \{\xi\mapsto{}z(t,\xi;\mu):t\in{}I\mu\in\mathbb{P}}$.

## General workflow

In reduced order modelling we aim to construct a mapping to a space that is close to this solution manifold. This is done through the following steps: 

1. Discretize the PDE.

2. Solve the discretized PDE for a certain set of parameter instances $\mu\in\mathbb{P}$.

3. Build a reduced basis with the data obtained from having solved the discretized PDE. This step consists of finding two mappings: the **reduction** $\mathcal{P}$ and the **reconstruction** $\mathcal{R}$.

