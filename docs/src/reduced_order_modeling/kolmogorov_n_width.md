# Kolmogorov $n$-width

The Kolmogorov $n$-width measures how well some set $\mathcal{M}$ (typically the solution manifold) can be approximated with a linear subspace:

```math
d_n(\mathcal{M}) := \mathrm{inf}_{V_n\subset{}V;\mathrm{dim}V_n=n}\mathrm{sup}(u\in\mathcal{M})\mathrm{inf}_{v_n\in{}V_n}|| u - v_n ||_V,
```

with $\mathcal{M}\subset{}V$ and $V$ is a (typically infinite-dimensional) Banach space. For advection-dominated problems (among others) the **decay of the Kolmogorov $n$-width is very slow**, i.e. one has to pick $n$ very high in order to obtain useful approximations (see [greif2019decay](@cite) and [blickhan2023registration](@cite)).

In order to overcome this, techniques based on neural networks (see e.g. [lee2020model](@cite)) and optimal transport (see e.g. [blickhan2023registration](@cite)) have been used. 


## References 

```@bibliography
Pages = []
Canonical = false 

blickhan2023registration
greif2019decay
lee2020model
```