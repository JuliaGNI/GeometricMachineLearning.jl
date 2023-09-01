# Optimizer

In order to generalize neural network optimizers to **homogeneous spaces**, a class of manifolds we often encounter in machine learning, we have to find a [global tangent space representation](arrays/stiefel_lie_alg_horizontal.md) which we call $\mathfrak{g}^\mathrm{hor}$ here. 

Starting from an element of the tangent space $T_Y\mathcal{M}$, we need to perform two mappings to arrive at $\mathfrak{g}^\mathrm{hor}$, illustrated in the following:

![](images/general_optimization.png)

Here the mapping $\Omega$ is a **horizontal lift** from the tangent space onto the **horizontal component of the Lie algebra at $Y$**. 