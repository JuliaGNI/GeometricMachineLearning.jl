# Optimizer

In order to generalize neural network optimizers to [homogeneous spaces](manifolds/homogeneous_spaces.md), a class of manifolds we often encounter in machine learning, we have to find a [global tangent space representation](arrays/stiefel_lie_alg_horizontal.md) which we call $\mathfrak{g}^\mathrm{hor}$ here. 

Starting from an element of the tangent space $T_Y\mathcal{M}$[^1], we need to perform two mappings to arrive at $\mathfrak{g}^\mathrm{hor}$, which we refer to by $\Omega$ and a red horizontal arrow:

[^1]: In practice this is obtained by first using an AD routine on a loss function $L$, and then computing the Riemannian gradient based on this. See the section of the [Stiefel manifold](manifolds/stiefel_manifold.md) for an example of this.

```@example
import Images, Plots # hide
if Main.output_type == :html # hide
    HTML("""<object type="image/svg+xml" class="display-light-only" data=$(joinpath(Main.buildpath, "tikz/general_optimization_with_boundary.png"))></object>""") # hide
else # hide
  Plots.plot(Images.load("tikz/general_optimization_with_boundary.png"), axis=([], false)) # hide
end # hide
```

```@example
if Main.output_type == :html # hide
    HTML("""<object type="image/svg+xml" class="display-dark-only" data=$(joinpath(Main.buildpath, "tikz/general_optimization_with_boundary_dark.png"))></object>""") # hide
end # hide
```

Here the mapping $\Omega$ is a [horizontal lift](optimizers/manifold_related/horizontal_lift.md) from the tangent space onto the **horizontal component of the Lie algebra at $Y$**. 

The red line maps the horizontal component at $Y$, i.e. $\mathfrak{g}^{\mathrm{hor},Y}$, to the horizontal component at $\mathfrak{g}^\mathrm{hor}$.

The $\mathrm{cache}$ stores information about previous optimization steps and is dependent on the optimizer. The elements of the $\mathrm{cache}$ are also in $\mathfrak{g}^\mathrm{hor}$. Based on this the optimer ([Adam](optimizers/adam_optimizer.md) in this case) computes a final velocity, which is the input of a [retraction](optimizers/manifold_related/retractions.md). Because this *update* is done for $\mathfrak{g}^{\mathrm{hor}}\equiv{}T_Y\mathcal{M}$, we still need to perform a mapping, called `apply_section` here, that then finally updates the network parameters. The two red lines are described in [global sections](optimizers/manifold_related/global_sections.md).

## References 

```@bibliography
Pages = []
Canonical = false

brantner2023generalizing
```