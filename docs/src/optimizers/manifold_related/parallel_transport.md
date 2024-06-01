# Parallel Transport

The concept of *parallel transport along a geodesic* ``\gamma:[0, T]\to\mathcal{M}`` describes moving a tangent vector from ``T_x\mathcal{M}`` to ``T_{\gamma(t)}\mathcal{M}`` such that its orientation with respect to the geodesic is preserved.

A precise definition of parallel transport needs a notion of a *connection* [lang2012fundamentals, bishop1980tensor, bendokat2020grassmann](@cite). Here we simply state how to parallel transport vectors on the Lie group ``SO(N)`` and the homogeneous spaces ``St(n, N)`` and ``Gr(n, N)``.

```@eval
Main.theorem(raw"Given two elements ``B^A_1, B^A_2\in{}T_AG`` the parallel transport of ``B^A_2`` along the geodesic of ``B^A_1`` is given by
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Pi_{A\to\gamma_{B^A_1}(t)} = A\exp(t\cdot{}A^{-1}B^A_1)A^{-1}B^A_2 = A\exp(t\cdot{}B_1)B_2,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``B_i := A^{-1}B_i.``")
```

For the Stiefel manifold this is not much more complicated[^1]:

[^1]: That this expression is sound from the perspective of Riemannian geometry has to be proved [schlarb2024covariant](@cite). For now the evidence that this is correct is largely empirical. 

```@eval
Main.theorem(raw"Given two elements ``\Delta_1, \Delta_2\in{}T_Y\mathcal{M}``, the parallel transport of ``\Delta_2`` along the geodesic of ``\Delta_1`` is given by
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Pi_{Y\to\gamma_{\Delta_1}(t)} = \exp(t\cdot\Omega(Y, \Delta_1))\Delta_2 =  \lambda(Y)\exp(B_1)\lambda(Y)^{-1}\Delta_2,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``B_1 = \lambda(Y)^{-1}\Omega(Y, \Delta_1)\lambda(Y).``")
```

We can further modify the expression of parallel transport for the Stiefel manifold: 

```math
\Pi_{Y\to\gamma_{\Delta_1}(t)} = \lambda(Y)\exp(B_1)\lambda(Y)\Omega(Y, \Delta_2)Y = \lambda(Y)\exp(B_1)B_2E,
```

where ``B_2 = \lambda(Y)^{-1}\Omega(Y, \Delta_2)\lambda(Y).``. We can now define explicit updating rules for the global section ``\Lambda^{(\cdot)}``, the element of the homogeneous space ``Y^{(\cdot)}``, the tangent vector ``\Delta^{(\cdot)}`` and ``D^{(\cdot)}``, its representation in ``\mathfrak{g}^\mathrm{hor}``.

We thus have:
1. ``\Lambda^{(t)} \leftarrow \Lambda^{(t-1)}\exp(B^{(t-1)}),``
2. ``Y^{(t)} \leftarrow \Lambda^{(t)}E,``
3. ``\Delta^{(t)} \leftarrow  \Lambda^{(t-1)}\exp(B^{(t-1)})(\Lambda^{(t-1)})^{-1}\Delta^{(t-1)} = \Lambda^{(t)}D^{(t-1)}E,``
4. ``D^{(t)} \leftarrow D^{(t-1)}.``

So we conveniently take parallel transport of vectors into account by representing them in ``\mathfrak{g}^\mathrm{hor}``.

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

v = 2 * rand(3, 1)
v₂ = 1 * rand(3, 1)
Δ = rgrad(Y, v)
Δ₂ = rgrad(Y, v₂)

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)

function set_up_plot(; theme = :dark) # hide
text_color = Main.output_type == :html ? :white : :black # hide
fig = Figure(; backgroundcolor = :transparent) # hide
text_color = Main.output_type == :html ? :white : :black # hide
ax = Axis3(fig[1, 1]; # hide
        backgroundcolor = (:tomato, .5), # hide
        aspect = (1., 1., 1.), # hide
        azimuth = π / 6, # hide
        elevation = π / 8, # hide
        xlabel = rich("x", subscript("1"), font = :italic, color = text_color), # hide
        ylabel = rich("x", subscript("2"), font = :italic, color = text_color), # hide
        zlabel = rich("x", subscript("3"), font = :italic, color = text_color), # hide
        ) # hide

# plot a sphere with radius one and origin 0
surface!(ax, Main.sphere(1., [0., 0., 0.])...; alpha = .5, transparency = true)

point_vec = ([Y[1]], [Y[2]], [Y[3]])
scatter!(ax, point_vec...; color = morange, marker = :star5)

arrow_vec = ([Δ[1]], [Δ[2]], [Δ[3]])
arrows!(ax, point_vec..., arrow_vec...; color = mred, linewidth = .02)

arrow_vec2 = ([Δ₂[1]], [Δ₂[2]], [Δ₂[3]])
arrows!(ax, point_vec..., arrow_vec2...; color = mpurple, linewidth = .02)

fig, ax # hide
end # hide

fig_light = set_up_plot(; theme = :light)[1]
fig_dark = set_up_plot(; theme = :dark)[1]
save("two_vectors.png", fig_light |> alpha_colorbuffer) # hide
save("two_vectors_dark.png", fig_dark |> alpha_colorbuffer) # hide

nothing # hide
```

```@example
Main.include_graphics("two_vectors") # hide
```

Note that we have chosen the arrow here to have the same direction as before but only about half the magnitude. We further drew another arrow that we want to parallel transport. 

```@example s2_parallel_transport
using GeometricMachineLearning: update_section!
λY = GlobalSection(Y)
B = global_rep(λY, Δ)
B₂ = global_rep(λY, Δ₂)

E = StiefelProjection(3, 1)
Y_increments = []
Δ_transported = []
Δ₂_transported = []

const n_steps = 8
const tstep = 2

for _ in 1:n_steps
    update_section!(λY, tstep * B, geodesic)
    push!(Y_increments, copy(λY.Y))
    push!(Δ_transported, Matrix(λY) * B * E)
    push!(Δ₂_transported, Matrix(λY) * B₂ * E)
end

function plot_parallel_transport(; theme = :dark) # hide
fig, ax = set_up_plot(; theme = :dark) # hide
for Y_increment in Y_increments
    scatter!(ax, [Y_increment[1]], [Y_increment[2]], [Y_increment[3]]; 
        color = mred, markersize = 5)
end

for (color, vec_transported) in zip((mred, mpurple), (Δ_transported, Δ₂_transported))
    for (Y_increment, vec_increment) in zip(Y_increments, vec_transported)
        point_vec = ([Y_increment[1]], [Y_increment[2]], [Y_increment[3]])
        arrow_vec = ([vec_increment[1]], [vec_increment[2]], [vec_increment[3]])
        arrows!(ax, point_vec..., arrow_vec...; color = color, linewidth = .02) 
    end
end

fig
end # hide

fig_light = plot_parallel_transport(; theme = :light) # hide
fig_dark = plot_parallel_transport(; theme = :dark) # hide
save("parallel_transport.png", fig_light |> alpha_colorbuffer) # hide
save("parallel_transport_dark.png", fig_dark |> alpha_colorbuffer) # hide

nothing # hide
```

```@example
Main.include_graphics("parallel_transport") # hide
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