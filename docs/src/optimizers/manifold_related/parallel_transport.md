# Parallel Transport

The concept of *parallel transport along a geodesic* ``\gamma:[0, T]\to\mathcal{M}`` describes moving a tangent vector from ``T_x\mathcal{M}`` to ``T_{\gamma(t)}\mathcal{M}`` such that its orientation with respect to the geodesic is preserved.

A precise definition of parallel transport needs a notion of a *connection* [lang2012fundamentals, bishop1980tensor, bendokat2020grassmann](@cite) and we forego it here. We simply state how to parallel transport vectors on the Lie group ``SO(N)`` and the homogeneous spaces ``St(n, N)`` and ``Gr(n, N)``.

```@eval
Main.theorem(raw"Given two elements ``B^A_1, B^A_2\in{}T_AG`` the parallel transport of ``B^A_2`` along the geodesic of ``B^A_1`` is given by
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Pi_{A\to\gamma_{B^A_1}(t)}B^A_2 = A\exp(t\cdot{}A^{-1}B^A_1)A^{-1}B^A_2 = A\exp(t\cdot{}B_1)B_2,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``B_i := A^{-1}B^A_i.``")
```

For the Stiefel manifold this is not much more complicated[^1]:

[^1]: Here we do not provide a detailed proof that this constitutes a sound expression from the perspective of Riemannian geometry. A proof can be found in [schlarb2024covariant](@cite).

```@eval
Main.theorem(raw"Given two elements ``\Delta_1, \Delta_2\in{}T_Y\mathcal{M}``, the parallel transport of ``\Delta_2`` along the geodesic of ``\Delta_1`` is given by
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Pi_{Y\to\gamma_{\Delta_1}(t)}\Delta_2 = \exp(t\cdot\Omega(Y, \Delta_1))\Delta_2 =  \lambda(Y)\exp(\bar{B}_1)\lambda(Y)^{-1}\Delta_2,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\bar{B}_1 = \lambda(Y)^{-1}\Omega(Y, \Delta_1)\lambda(Y).``")
```

We can further modify the expression of parallel transport for the Stiefel manifold: 

```math
\Pi_{Y\to\gamma_{\Delta_1}(t)}\Delta_2 = \lambda(Y)\exp(B_1)\lambda(Y)\Omega(Y, \Delta_2)Y = \lambda(Y)\exp(B_1)B_2E,
```

where ``B_2 = \lambda(Y)^{-1}\Omega(Y, \Delta_2)\lambda(Y).`` We can now define explicit updating rules for the [global section](@ref "Global Sections") ``\Lambda^{(\cdot)}``, the element of the homogeneous space ``Y^{(\cdot)}``, the tangent vector ``\Delta^{(\cdot)}`` and ``D^{(\cdot)} = (\Lambda^{(\cdot)})^{-1}\Omega(\Delta^{(\cdot)})\Lambda^{(\cdot)}``, its representation in ``\mathfrak{g}^\mathrm{hor}``.

We thus have:
1. ``\Lambda^{(t)} \leftarrow \Lambda^{(t-1)}\exp(B^{(t-1)}),``
2. ``Y^{(t)} \leftarrow \Lambda^{(t)}E,``
3. ``\Delta^{(t)} \leftarrow  \Lambda^{(t-1)}\exp(B^{(t-1)})(\Lambda^{(t-1)})^{-1}\Delta^{(t-1)} = \Lambda^{(t)}D^{(t-1)}E,``
4. ``D^{(t)} \leftarrow D^{(t-1)}.``

So we conveniently take parallel transport of vectors into account by representing them in ``\mathfrak{g}^\mathrm{hor}``: ``D`` does not change.

To demonstrate parallel transport we again use the example from when we introduced the concept of [geodesics](@ref "Geodesic Sprays and the Exponential Map"). We first set up the problem:

```@setup s2_parallel_transport
using GLMakie

include("../../../gl_makie_transparent_background_hack.jl")
```

```@setup s2_parallel_transport
using GeometricMachineLearning
import Random # hide
Random.seed!(123) # hide

Y = rand(StiefelManifold, 3, 1)
# needed because we will change `Y` later on
Y_copy = StiefelManifold(copy(Y.A))

v = 2 * rand(3, 1)
v₂ = 1 * rand(3, 1)
Δ = rgrad(Y, v)
Δ₂ = rgrad(Y, v₂)

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide

function set_up_plot(; theme = :dark) # hide
fig = Figure(; backgroundcolor = :transparent, size = (900, 675)) # hide
text_color = theme == :dark ? :white : :black # hide
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
    limits = ([-1, 1], [-1, 1], [-1, 1]),
    azimuth = π / 7, # hide
    elevation = π / 7, # hide
    # height = 75.,
    ) # hide

# plot a sphere with radius one and origin 0
surface!(ax, Main.sphere(1., [0., 0., 0.])...; alpha = .5, transparency = true)

point_vec = ([Y_copy[1]], [Y_copy[2]], [Y_copy[3]])
scatter!(ax, point_vec...; color = morange, marker = :star5, markersize = 30)

arrow_vec = ([Δ[1]], [Δ[2]], [Δ[3]])
arrows!(ax, point_vec..., arrow_vec...; color = mred, linewidth = .02)

arrow_vec2 = ([Δ₂[1]], [Δ₂[2]], [Δ₂[3]])
arrows!(ax, point_vec..., arrow_vec2...; color = mpurple, linewidth = .02)

fig, ax # hide
end # hide

fig_light = set_up_plot(; theme = :light)[1]
fig_dark = set_up_plot(; theme = :dark)[1]
save("two_vectors.png", alpha_colorbuffer(fig_light)) # hide
save("two_vectors_dark.png", alpha_colorbuffer(fig_dark)) # hide

nothing # hide
```

![](two_vectors_light.png)
![](two_vectors_dark.png)

Note that we have chosen the arrow here to have the same direction as before but only about half the magnitude. We further drew another arrow that we want to parallel transport (the purple arrow). 

```@example s2_parallel_transport
using GeometricMachineLearning: update_section! # hide

λY = GlobalSection(Y)
B = global_rep(λY, Δ)
B₂ = global_rep(λY, Δ₂)

E = StiefelProjection(3, 1)
Y_increments = []
Δ_transported = []
Δ₂_transported = []

const n_steps = 6
const tstep = 2

for _ in 1:n_steps
    update_section!(λY, tstep * B, geodesic)
    push!(Y_increments, copy(λY.Y))
    push!(Δ_transported, Matrix(λY) * B * E)
    push!(Δ₂_transported, Matrix(λY) * B₂ * E)
end
nothing # hide
```

```@setup s2_parallel_transport
function plot_parallel_transport(; theme = :dark) # hide
fig, ax = set_up_plot(; theme = theme) # hide
for Y_increment in Y_increments 
    scatter!(ax, [Y_increment[1]], [Y_increment[2]], [Y_increment[3]]; 
        color = mred)
end

for (color, vec_transported) in zip((mred, mpurple), (Δ_transported, Δ₂_transported))
    for (Y_increment, vec_increment) in zip(Y_increments, vec_transported)
        point_vec = ([Y_increment[1]], [Y_increment[2]], [Y_increment[3]])
        arrow_vec = ([vec_increment[1]], [vec_increment[2]], [vec_increment[3]])
        arrows!(ax, point_vec..., arrow_vec...; color = color, linewidth = .02) 
    end
end

fig, ax
end # hide

fig_light, ax_light = plot_parallel_transport(; theme = :light) # hide
fig_dark, ax_dark = plot_parallel_transport(; theme = :dark) # hide
save("parallel_transport.png", fig_light |> alpha_colorbuffer) # hide
save("parallel_transport_dark.png", fig_dark |> alpha_colorbuffer) # hide
hidedecorations!(ax_light)  # hide
hidespines!(ax_light) # hide
save("parallel_transport_naked.png", fig_light |> alpha_colorbuffer) # hide

nothing # hide
```

![](parallel_transport_light.png)
![](parallel_transport_dark.png)

Note that the angle between the two vector is preserved as we go along the geodesic.


```@raw latex
\section*{Chapter Summary}

In this chapter we introduced our \textit{optimizer framework} which will be used to efficiently train symplectic autoencoders and transformers with orthogonality constraints in Part IV. We proposed extending standard neural network optimizers to homogeneous spaces by introducing the extra operations \texttt{rgrad}, \texttt{global\_rep} and ``Retraction.'' The definition of a retraction we used here was slightly different from the usual one. We defined retractions as maps from the \textit{global tangent space representation} $\mathfrak{g}^\mathrm{hor}$ to the associated Lie group (and in addition satisfy two more conditions), i.e.
\begin{equation*}
    \mathrm{Retraction}: \mathfrak{g}^\mathrm{hor} \to G.
\end{equation*}

We further discussed what the operations \texttt{rgrad}, \texttt{global\_rep} and ``Retraction'' look like in practice and concluded by introducing the concept of \textit{parallel transport}. The presentation was accompanied by code snippets that demonstrate the application interface of \texttt{GeometricMachineLearning} throughout the chapter.

\begin{comment}
```

## References

```@bibliography
Pages = []
Canonical = false

lang2012fundamentals
bishop1980tensor
bendokat2020grassmann
schlarb2024covariant
```

```@raw latex
\end{comment}
```

```@raw html
<!--
```

# References

```@bibliography
Canonical = false
Pages = []

brantner2023generalizing
schlarb2024covariant
absil2008optimization
gao2021riemannian
gao2024optimization
bendokat2020grassmann
bendokat2021real
```

```@raw html
-->
```