# Riemannian Manifolds

A Riemannian manifold is a manifold ``\mathcal{M}`` that we endow with a mapping ``g`` that smoothly[^1] assigns a [metric](@ref "(Topological) Metric Spaces") ``g_x`` to each tangent space ``T_x\mathcal{M}``. By a slight abuse of notation we will also refer to this ``g`` as a *metric*.

[^1]: Smooth here refers to the fact that ``g:\mathcal{M}\to\text{(Space of Metrics)}`` has to be a smooth map. But in order to discuss this in detail we would have to define a topology on the space of metrics. A more detailed discussion can be found in [lang2012fundamentals, bishop1980tensor, do1992riemannian](@cite).

After having defined a metric ``g`` we can *associate a length* to each curve ``\gamma:[0, t] \to \mathcal{M}`` through: 

```math
L(\gamma) = \int_0^t \sqrt{g_{\gamma(s)}(\gamma'(s), \gamma'(s))}ds.
```

This ``L`` turns ``\mathcal{M}`` into a metric space:

```@eval
Main.definition(raw"The **metric on a Riemannian manifold** ``\mathcal{M}`` is 
" * 
Main.indentation * raw"```math
" *
Main.indentation * raw"d(x, y) = \inf_{\substack{\text{$\gamma(0) = x$ and}\\
    \gamma(t) = y}}L(\gamma),
" * 
Main.indentation * raw"```
" *
Main.indentation * raw"where ``t`` can be chosen arbitrarily.")
```

If a curve is minimal with respect to the function ``L`` we call it the *shortest curve* or a geodesic. So we say that a curve ``\gamma:[0, t]\to\mathcal{M}`` is a geodesic if there is no shorter curve that can connect two points in ``\gamma([0, t])``, i.e. 

```math
d(\gamma(t_i), \gamma(t_f)) = \int_{t_i}^{t_f}\sqrt{g_{\gamma(s)}(\gamma'(s), \gamma'(s))}ds,
```
for any ``t_i, t_f\in[0, t]``.

An important result of Riemannian geometry states that there exists a vector field ``X`` on ``T\mathcal{M}``, called the *geodesic spray*, whose integral curves are derivatives of geodesics. We formalize this statement as a theorem in the next section.


## Geodesic Sprays and the Exponential Map

To every Riemannian manifold we can naturally associate a vector field called the *geodesic spray* or *geodesic equation*. For our purposes it is enough to state that this vector field is unique and well-defined [do1992riemannian](@cite).

The important property of the geodesic spray is

```@eval
Main.theorem(raw"Given an initial point ``x`` and an initial velocity ``v_x``, an integral curve for the geodesic spray is of the form ``t \mapsto (\gamma_{v_x}(t), \gamma_{v_x}'(t))`` where ``\gamma_{v_x}`` is a geodesic. We further have the property that the integral curve for the geodesic spray for an initial point ``x`` and an initial velocity ``\eta\cdot{}v_x`` (where ``\eta`` is a scalar) is of the form ``t \mapsto (\gamma_{\eta\cdot{}v_x}(t), \gamma_{\eta\cdot{}v_x}'(t)) = (\gamma_{v_x}(\eta\cdot{}t), \eta\cdot\gamma_{v_x}'(\eta\cdot{}t)).``")
```

It is therefore customary to introduce the *exponential map* ``\exp:T_x\mathcal{M}\to\mathcal{M}`` as

```math
\exp(v_x) := \gamma_{v_x}(1),
```

and we see that ``\gamma_{v_x}(t) = \exp(t\cdot{}v_x)``. In `GeometricMachineLearning` we denote the exponential map by [`geodesic`](@ref) to avoid confusion with the matrix exponential map[^2] which is called as `exp` in Julia. So we use the definition:

[^2]: The Riemannian exponential map and the matrix exponential map coincide for many matrix Lie groups.

```math
    \mathtt{geodesic}(x, v_x) \equiv \exp(v_x).
```

We give an example of using this function here:

```@setup s2_retraction
using GLMakie

include("../../gl_makie_transparent_background_hack.jl")
```

```@example s2_retraction
using GeometricMachineLearning # hide
import Random # hide
Random.seed!(123) # hide

Y = rand(StiefelManifold, 3, 1)

v = 5 * rand(3, 1)
Δ = v - Y * (v' * Y)

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
function set_up_plot(; theme = :dark) # hide
text_color = theme == :dark ? :white : :black # hide
fig = Figure(; backgroundcolor = :transparent, size = (900, 675)) # hide
ax = Axis3(fig[1, 1]; # hide
    backgroundcolor = (:tomato, .5), # hide
    aspect = (1., 1., 1.), # hide
    xlabel = L"x_1", # hide
    ylabel = L"x_2", # hide
    zlabel = L"x_3", # hide
    xgridcolor = text_color, # hide
    ygridcolor = text_color, # hide
    zgridcolor = text_color, # hide
    xtickcolor = text_color, # hide
    ytickcolor = text_color, # hide
    ztickcolor = text_color, # hide
    xlabelcolor = text_color, # hide
    ylabelcolor = text_color, # hide
    zlabelcolor = text_color, # hide
    xypanelcolor = :transparent, # hide
    xzpanelcolor = :transparent, # hide
    yzpanelcolor = :transparent, # hide
    limits = ([-1, 1], [-1, 1], [-1, 1]), # hide
    azimuth = π / 7, # hide
    elevation = π / 7, # hide
    # height = 75., # hide
    ) # hide
# plot a sphere with radius one and origin 0
surface!(ax, Main.sphere(1., [0., 0., 0.])...; alpha = .5, transparency = true)

point_vec = ([Y[1]], [Y[2]], [Y[3]])
scatter!(ax, point_vec...; color = morange, marker = :star5, markersize = 30)

arrow_vec = ([Δ[1]], [Δ[2]], [Δ[3]])
arrows!(ax, point_vec..., arrow_vec...; color = mred, linewidth = .02)

fig, ax # hide
end # hide

fig_light = set_up_plot(; theme = :light)[1] # hide
fig_dark = set_up_plot(; theme = :dark)[1] # hide

save("sphere_with_tangent_vec.png", alpha_colorbuffer(fig_light)) # hide
save("sphere_with_tangent_vec_dark.png", alpha_colorbuffer(fig_dark)) # hide

nothing # hide
```

```@example
Main.include_graphics("sphere_with_tangent_vec"; width = .7) # hide
```

We now solve the geodesic spray for ``\eta\cdot\Delta`` for ``\eta = 0.1, 0.2, \ldots, 5.5`` with the function [`geodesic`](@ref) and plot the corresponding points:

```@example s2_retraction
Δ_increments = [Δ * η for η in 0.1 : 0.1 : 5.5]

Y_increments = [geodesic(Y, Δ_increment) for Δ_increment in Δ_increments]

function make_plot_with_solution(; theme = :dark) # hide
fig, ax = set_up_plot(; theme = theme) # hide
for Y_increment in Y_increments
    scatter!(ax, [Y_increment[1]], [Y_increment[2]], [Y_increment[3]]; 
        color = mred)
end

fig # hide
end # hide

fig_light = make_plot_with_solution(; theme = :light) # hide
fig_dark = make_plot_with_solution(; theme = :dark) # hide

save("sphere_with_tangent_vec_and_geodesic.png", alpha_colorbuffer(fig_light)) # hide
save("sphere_with_tangent_vec_and_geodesic_dark.png", alpha_colorbuffer(fig_dark)) # hide

nothing # hide
```

```@example
Main.include_graphics("sphere_with_tangent_vec_and_geodesic"; width = .7) # hide
```

A geodesic can be seen as the *equivalent of a straight line* on a manifold. Also note that we drew a random element form [`StiefelManifold`](@ref) here, and not from ``S^2``. This is because the category of [Stiefel manifolds](@ref "The Stiefel Manifold") is more general than the category of spheres ``S^n``: ``St(1, 3) \simeq S^2``.

## The Riemannian Gradient

The *Riemannian gradient* is essential when talking about optimization on manifolds.

```@eval
Main.definition(raw"The Riemannian gradient of a function ``L:\mathcal{M}\to\mathbb{R}`` is a vector field ``\mathrm{grad}^gL`` (or simply ``\mathrm{grad}L``) for which we have
" * Main.indentation * raw"```math
" * Main.indentation * raw"    g_x(\mathrm{grad}^gL(x), v_x) = (\nabla_{\varphi_U(x)}(L\circ\varphi_U^{-1}))^T \varphi_U'(v_x), 
" * Main.indentation * raw"```
" * Main.indentation * raw"for all ``v_x\in{}T_x\mathcal{M}.`` In the expression above ``\varphi_U`` is some coordinate chart defined in a neighborhood ``U`` around ``x``.")
```

In the definition above ``\nabla`` indicates the *Euclidean gradient*:
```math
 \nabla_xf = \begin{pmatrix} \frac{\partial{}f}{\partial{}x_1} \\ \cdots \\ \frac{\partial{}f}{\partial{}x_n} \end{pmatrix}.
```

We can also describe the Riemannian gradient through differential curves:

```@eval
Main.definition(raw"The Riemannian gradient of ``L`` is a vector field ``\mathrm{grad}^gL`` for which
" * Main.indentation * raw"```math
" * Main.indentation * raw"g_x(\mathrm{grad}^gL(x), \dot{\gamma}(0)) = \frac{d}{dt}L(\gamma(t)),
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\gamma`` is a ``C^\infty`` curve through ``x``.")
```

By the *non degeneracy* of ``g`` the Riemannian gradient always exists [bishop1980tensor](@cite). In the following we will also write ``\mathrm{grad}^gL(x) = \mathrm{grad}^g_xL = \mathrm{grad}_xL.`` We will give specific examples of this when discussing the [Stiefel manifold](@ref "The Stiefel Manifold") and the [Grassmann manifold](@ref "The Grassmann Manifold"). 


## Gradient Flows and Riemannian Optimization

In `GeometricMachineLearning` we can include weights in neural networks that are part of a manifold. Training such neural networks amounts to *Riemannian optimization* and hence solving the *gradient flow* equation. The gradient flow equation is given by

```math
X(x) = - \mathrm{grad}_xL.
```

Solving this gradient flow equation will then lead us to a local minimum on ``\mathcal{M}``. This will be elaborated on when talking about [optimizers](@ref "Neural Network Optimizers"). In practice we cannot solve the gradient flow equation directly and have to rely on approximations. The most straightforward approximation (and one that serves as a basis for all the optimization algorithms in `GeometricMachineLearning`) is to take the point ``(x, X(x))`` as an initial condition for the geodesic spray and then solve the ODE for a small time step. Such an update rule, i.e. 

```math
x^{(t)} \leftarrow \gamma_{X(x^{(t-1)})}(\Delta{}t)\text{ with $\Delta{}t$ the time step},
```

we call the *gradient optimization scheme*.

## Library Functions

```@docs
geodesic(::Manifold{T}, ::AbstractMatrix{T}) where T
```

## References

```@bibliography
Pages = []
Canonical = false

lang2012fundamentals
do1992riemannian
```