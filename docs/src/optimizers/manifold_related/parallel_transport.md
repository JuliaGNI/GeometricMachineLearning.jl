# Parallel Transport

The concept of *parallel transport along a geodesic* ``\gamma:[0, T]\to\mathcal{M}`` describes moving a tangent vector from ``T_x\mathcal{M}`` to ``T_{\gamma(t)}\mathcal{M}`` such that its orientation with respect to the geodesic is preserved.

```math
\Pi_{A \to \exp(V_A)}\tilde{V}_A = \exp(V_AA^{-1})\tilde{V}_A
```

```math
\Pi_{Y \to \gamma_\Delta(\eta)}\Delta_2 = \exp(\Omega(Y, \Delta))\Delta_2
```

We again use the example from when we introduced the concept of [geodesics](@ref "Geodesic Sprays and the Exponential Map").

```@example s2_setup
using GeometricMachineLearning
using CairoMakie # hide
import Random # hide
Random.seed!(123) # hide

Y = rand(StiefelManifold, 3, 1)

v = 5 * rand(3, 1)
v₂ = 5 * rand(3, 1)
Δ = rgrad(Y, v)
Δ₂ = rgrad(Y, v₂)

fig = Figure(; backgroundcolor = :transparent) # hide
text_color = Main.output_type == :html ? :white : :black # hide
ax = Axis3(fig[1, 1]; # hide
        backgroundcolor = :transparent, # hide
        aspect = (1., 1., 1.), # hide
        azimuth = π / 6, # hide
        elevation = π / 8, # hide
        xlabel = rich("x", subscript("1"), font = :italic, color = text_color), # hide
        ylabel = rich("x", subscript("2"), font = :italic, color = text_color), # hide
        zlabel = rich("x", subscript("3"), font = :italic, color = text_color), # hide
        ) # hide

# plot a sphere with radius one and origin 0
surface!(ax, Main.sphere(1., [0., 0., 0.])...; alpha = .6)

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
point_vec = ([Y[1]], [Y[2]], [Y[3]])
scatter!(ax, point_vec...; color = morange, marker = :star5)

mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
arrow_vec = ([Δ[1]], [Δ[2]], [Δ[3]])
arrows!(ax, point_vec..., arrow_vec...; color = mred, linewidth = .02)

mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
arrow_vec2 = ([Δ₂[1]], [Δ₂[2]], [Δ₂[3]])
arrows!(ax, point_vec..., arrow_vec2...; color = mpurple, linewidth = .02)

fig
```

```@example s2_retraction
Δ_increments = [Δ * η for η in 0.1 : 0.1 : 2.5]
λY = GlobalSection(Y)

B_increments = [global_rep(λY, Δ_increment) for Δ_increment in Δ_increments]

... define parallel transport!!!
A_increments = [geodesic(B_increment) for B_increment in B_increments]
Y_increments = [apply_section(λY, A_increment) for A_increment in ]

for Y_increment in Y_increments
    scatter!(ax, [Y_increment[1]], [Y_increment[2]], [Y_increment[3]]; 
        color = mred, markersize = 5)
end

fig
```