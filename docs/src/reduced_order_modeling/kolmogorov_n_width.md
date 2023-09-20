# Kolmogorov $n$-width

The Kolmogorov $n$-width measures how well some set $\mathcal{M}$ (typically the solution manifold) can be approximated with a linear subspace:

```math
d_n(\mathcal{M}) := \mathrm{inf}_{V_n\sub{}V;\mathrm{dim}V_n=n}\mathrm{sup}(u\in\mathcal{M})\mathrm{inf}_{v_n\in{}V_n}|| u - v_n ||_V,
```

with $\mathcal{M}\sub{}V$ and $V$ is a (typically infinite-dimensional) Banach space. For advection-dominated problems (among others) the **decay of the Kolmogorov $n$-width is very slow**, i.e. one has to pick $n$ very high in order to obtain useful approximations (see (Greif and Urban, 2019) and (Blickhan, 2023)).

In order to overcome this, techniques based on neural networks (see e.g. (Lee and Carlberg, 2020)) and optimal transport (see e.g. (Blickhan, 2023)) have been used. 


## References 
- Greif, Constantin, and Karsten Urban. "Decay of the Kolmogorov N-width for wave problems." Applied Mathematics Letters 96 (2019): 216-222.
- Blickhan, Tobias. "A registration method for reduced basis problems using linear optimal transport." arXiv preprint arXiv:2304.14884 (2023).
- Lee, Kookjin, and Kevin T. Carlberg. "Model reduction of dynamical systems on nonlinear manifolds using deep convolutional autoencoders." Journal of Computational Physics 404 (2020): 108973.
