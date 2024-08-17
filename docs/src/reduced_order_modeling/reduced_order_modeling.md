# Basic Concepts of Reduced Order Modeling

Reduced order modeling is a data-driven technique that exploits the structure of parametric partial differential equations (PPDEs) to make repeated simulations of this PPDE much cheaper.

For this consider a PPDE written in the form: ``F(z(\mu);\mu)=0`` where ``z(\mu)`` evolves on an infinite-dimensional Hilbert space ``V``. 

In modeling any PDE we have to choose a discretization (particle discretization, finite element method, ...) of ``V``, which will be denoted by ``V_h \simeq \mathbb{R}^N``. The space ``V_h`` is not infinite-dimensional but its dimension ``N`` is still very large. Solving a discretized PDE in this space is typically very expensive. In reduced order modeling we utilize the fact that slightly different choices of parameters ``\mu`` will give qualitatively similar solutions. We can therefore perform a few simulations in the full space ``V_h`` and then make successive simulations cheaper by *learning* from the past simulations. A crucial concept in this is the *solution manifold*.

## The Solution Manifold 

To any PPDE and a certain parameter set ``\mathbb{P}`` we associate a *solution manifold*: 

```math 
\mathcal{M} = \{z(\mu):F(z(\mu);\mu)=0, \mu\in\mathbb{P}\}.
```

A motivation for reduced order modeling is that even though the space ``V_h`` is of very high-dimension, the solution manifold will typically be a very small space. The image below shows a two-dimensional solution manifold[^1] embedded in ``V_h\equiv\mathbb{R}^3``:

[^1]: The systems be deal with usually have much greater dimension of course. The dimension of ``V_h`` will be in the thousands and the dimension of the solution manifold will be a few orders of magnitudes smaller. Because this cannot be easily visualized, we resort to showing a two-dimensional manifold in a three-dimensional space here. 

![](../tikz/solution_manifold_2.png)

As an actual example of a solution manifold consider the one-dimensional wave equation [blickhan2023registration](@cite): 

```math
\partial_{tt}^2q(t,\omega;\mu) = \mu^2\partial_{\omega\omega}^2q(t,\omega;\mu)\text{ on }I\times\Omega,
```
where ``I = (0,1)`` and ``\Omega=(-1/2,1/2)``. As initial condition for the first derivative we have ``\partial_tq(0,\omega;\mu) = -\mu\partial_\omega{}q_0(\xi;\mu)`` and furthermore ``q(t,\omega;\mu)=0`` on the boundary (i.e. ``\omega\in\{-1/2,1/2\}``).

The solution manifold is a two-dimensional submanifold of an infinite-dimensional function space: 

```math
\mathcal{M} = \{(t, \omega)\mapsto{}q(t,\omega;\mu)=q_0(\omega-\mu{}t;\mu):\mu\in\mathbb{P}\subset\mathbb{R}\}.
```

We can plot some of the *points* on ``\mathcal{M}`` (each curve correspond to one point): 

```@eval
using CairoMakie
import GeometricProblems.LinearWave as lw

# specify different μ values
μs = (0.416, 0.508, 0.6)
time_steps = (0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)

colors = (morange, mpurple, mred)

function make_plot(; theme = :dark, plot_name = "wave_equation_different_parameters")
fig = Figure(; backgroundcolor = :transparent)
text_color = theme == :dark ? :white : :black

function make_axis(i)
    μ = μs[i]
    ax = Axis(fig[i, 1]; 
        xlabel = L"\omega",
        ylabel = L"q(t, \omega, \mu)",
        xgridcolor = text_color,
        ygridcolor = text_color,
        xtickcolor = text_color,
        ytickcolor = text_color,
        xlabelcolor = text_color,
        ylabelcolor = text_color,
        backgroundcolor = :transparent)
    # plot 6 time steps
    domain = lw.compute_domain(lw.Ñ + 2)
    for time_step in time_steps
        lines!(ax, domain, lw.u₀(domain .- μ * time_step, μ), color = colors[i])
    end
    ax
end

ax1 = make_axis(1)
ax2 = make_axis(2)
ax3 = make_axis(3)

add_on = theme == :dark ? "_dark" : ""
save(plot_name * add_on * ".png", fig; px_per_unit = 1)
end

make_plot(; theme = :dark)
make_plot(; theme = :light)

nothing
```

```@example
Main.include_graphics("wave_equation_different_parameters") # hide
```

Here we plotted the curves for the time steps (0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0) and the parameter values (0.416, 0.508, 0.6). We see that, depending on the parameter value ``\mu``, the wave travels at different speeds. In reduced order modeling we try to find an approximation to the solution manifolds, i.e. model the evolution of the curve in a cheap way for different parameter values ``\mu``. Neural networks offer a way of doing so efficiently!

## General Workflow

In reduced order modeling we aim to construct an approximation to the solution manifold and that is ideally of a dimension not much greater than that of the solution manifold and then (approximately) solve the so-called *reduced equations* in the small space. Constructing this approximation to the solution manifold can be divided into three steps[^2]: 
1. Discretize the PDE, i.e. find ``V\to{}V_h``.
2. Solve the discretized PDE on ``V_h`` for a certain set of parameter instances ``\mu\in\mathbb{P}``.
3. Build a reduced basis with the data obtained from having solved the discretized PDE. This step consists of finding two mappings: the *reduction* ``\mathcal{P}`` and the *reconstruction* ``\mathcal{R}``.

[^2]: Approximating the solution manifold is referred to as the *offline phase* of reduced order modeling.

The third step can be done with various machine learning (ML) techniques. Traditionally the most popular of these has been *Proper orthogonal decomposition* (POD), but in recent years *autoencoders* have become a widely-used alternative [fresca2021comprehensive, lee2020model](@cite).

After having obtained ``\mathcal{P}`` and ``\mathcal{R}`` we still need to solve the *reduced system*. Solving the reduced system is typically referred to as the *online phase* in reduced order modeling. This is sketched below: 

```@example
Main.include_graphics("../tikz/offline_online") # hide
```

The online phase is applying the mapping ``\mathcal{NN}`` in the low-dimensional space in order to predict the next time step; this can either be done with a standard integrator [Kraus:2020:GeometricIntegrators](@cite) or, as is indicated here, [with a neural network](@ref "Neural Network Integrators"). Crucially this step can be made very cheap when compared to the full-order model[^3]. In the following we discuss how an equation for the reduced model can be found classically, without relying on a neural network for the online phase.

[^3]: Solving the reduced system is typically faster by a factor of at least ``10^3.``

## Obtaining the Reduced System via Galerkin Projection

*Galerkin projection* [gander2012euler](@cite) offers a way of constructing an ODE on the reduced space once the reconstruction ``\mathcal{R}`` has been found. 

```@eval
Main.definition(raw"Given a full-order model described by a differential equation ``F(\cdot; \mu):V\to{}V``, where ``V`` may be an infinite-dimensional Hilbert space (PDE case) or a finite-dimensional vector space ``\mathbb{R}^N`` (ODE case), and a reconstruction ``\mathcal{R}:\mathbb{R}^n\to{}V``, we can find an equation on the reduced space ``\mathbb{R}^n.`` For this we first take as possible solutions for the equation
" * Main.indentation * raw"```math
" * Main.indentation * raw"    F(\hat{u}(t); \mu) - \hat{u}'(t) = 0
" * Main.indentation * raw"```
" * Main.indentation * raw"the ones that are the *result of a reconstruction*:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    F(\mathcal{R}(u(t)); \mu) - d\mathcal{R}u'(t) = 0,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``u:[0, T]\to\mathbb{R}^n`` is an orbit on the reduced space and ``d\mathcal{R}`` is the differential of the reconstruction; this is ``\nabla{}\mathcal{R}`` if ``V`` is finite-dimensional. Typically we test this expression with a set of basis functions or vectors ``\{\tilde{\psi}_1, \ldots, \tilde{\psi}_n \}`` and hence obtain ``n`` scalar equations:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \langle F(\mathcal{R}(u(t)); \mu) - d\mathcal{R}u'(t), \psi_i \rangle_V \text{ for $1\leq{}i\leq{}n$}.
" * Main.indentation * raw"```
" * Main.indentation * raw"Such a procedure to obtain a reduced equation is known as **Galerkin projection**.")
```

We give specific examples of reduced systems obtained with a Galerkin projection when introducing [proper orthogonal decomposition](@ref "Proper Orthogonal Decomposition"), [autoencoders](@ref "Autoencoders"), [proper symplectic decomposition](@ref "Proper Symplectic Decomposition") and [symplectic auteoncoders](@ref "Symplectic Autoencoders").


## Kolmogorov ``n``-width

The Kolmogorov ``n``-width [arbes2023kolmogorov](@cite) measures how well some set ``\mathcal{M}`` (typically the solution manifold) can be approximated with a linear subspace:

```math
d_n(\mathcal{M}) := \mathrm{inf}_{V_n\subset{}V;\mathrm{dim}V_n=n}\mathrm{sup}(u\in\mathcal{M})\mathrm{inf}_{v_n\in{}V_n}|| u - v_n ||_V,
```

with ``\mathcal{M}\subset{}V`` and ``V`` is a (typically infinite-dimensional) Banach space. For advection-dominated problems (among others) the *decay of the Kolmogorov ``n``-width is very slow*, i.e. one has to pick ``n`` very high in order to obtain useful approximations (see [greif2019decay](@cite) and [blickhan2023registration](@cite)).

In order to overcome this, techniques based on neural networks (see e.g. [lee2020model](@cite)) and optimal transport (see e.g. [blickhan2023registration](@cite)) have been used. 

## References 
```@bibliography
Pages = []
Canonical = false

fresca2021comprehensive
lee2020model
blickhan2023registration
```