# Retractions

In practice we usually do not solve the geodesic equation exactly in each optimization step (even though this is possible and computationally feasible), but prefer approximations that are called "retractions" [absil2008optimization](@cite) for stability. The definition of a retraction in `GeometricMachineLearning` is slightly different from how it is usually defined in textbooks [absil2008optimization, hairer2006geometric](@cite). We discuss the differences here.

## Classical Retractions

By "classical retraction" we here mean the textbook definition. 

```@eval
Main.theorem(raw"A **classical retraction** is a smooth map
" * Main.indentation * raw"```math 
" * Main.indentation * raw"R: T\mathcal{M}\to\mathcal{M}:(x,v)\mapsto{}R_x(v),
" * Main.indentation * raw"```
" * Main.indentation * raw"such that each curve ``c(t) := R_x(tv)`` locally approximates the geodesic, i.e. the following two conditions hold:
" * Main.indentation * raw"1. ``c(0) = x`` and 
" * Main.indentation * raw"2. ``c'(0) = v.``
")
```

Perhaps the most common example for matrix manifolds is the *Cayley retraction*:

```@eval
Main.example(raw"The **Cayley retraction** is defined as
" * Main.indentation * raw"```math
" * Main.indentation * raw"\mathrm{Cayley}(V_x) = \left(\mathbb{I} - \frac{1}{2}V_x\right)^{-1}\left(\mathbb{I} +\frac{1}{2}V_x\right).
" * Main.indentation * raw"```")
```

We should mention that the factor ``\frac{1}{2}`` is sometimes left out in the definition of the Cayley transform. But if we leave this out we do not have a retraction. 

We want to compare the [`geodesic`](@ref) retraction with the [`cayley`](@ref) retraction for the example we already introduced when talking about the [exponential map](@ref "Geodesic Sprays and the Exponential Map"):

```@setup s2_retraction
using GeometricMachineLearning
using CairoMakie # hide
import Random # hide
Random.seed!(123) # hide

Y = rand(StiefelManifold, 3, 1)

v = 5 * rand(3, 1)
Δ = v - Y * (v' * Y)

function do_setup()
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

    fig, ax, point_vec
end

mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)

nothing
```

```@example s2_retraction
Δ_increments = [Δ * η for η in 0.1 : 0.1 : 2.5]

Y_increments_geodesic = [geodesic(Y, Δ_increment) for Δ_increment in Δ_increments]
Y_increments_cayley = [cayley(Y, Δ_increment) for Δ_increment in Δ_increments]
 # hide
function make_plot(; theme=:light) # hide

fig, ax, point_vec = do_setup() # hide

Y_zeros = zeros(length(Y_increments_geodesic))
Y_geodesic_reshaped = [copy(Y_zeros), copy(Y_zeros), copy(Y_zeros)]
Y_cayley_reshaped = [copy(Y_zeros), copy(Y_zeros), copy(Y_zeros)]

zip_ob = zip(Y_increments_geodesic, Y_increments_cayley, axes(Y_increments_geodesic, 1))

for (Y_increment_geodesic, Y_increment_cayley, i) in zip_ob
    Y_geodesic_reshaped[1][i] = Y_increment_geodesic[1]
    Y_geodesic_reshaped[2][i] = Y_increment_geodesic[2]
    Y_geodesic_reshaped[3][i] = Y_increment_geodesic[3]

    Y_cayley_reshaped[1][i] = Y_increment_cayley[1]
    Y_cayley_reshaped[2][i] = Y_increment_cayley[2]
    Y_cayley_reshaped[3][i] = Y_increment_cayley[3]
end

scatter!(ax, Y_geodesic_reshaped...; 
        color = mred, markersize = 5, label = "geodesic retraction")

scatter!(ax, Y_cayley_reshaped...; 
        color = mblue, markersize = 5, label = "Cayley retraction")

arrow_vec = ([Δ[1]], [Δ[2]], [Δ[3]]) # hide
arrows!(ax, point_vec..., arrow_vec...; color = mred, linewidth = .02) # hide
text_color = theme == :light ? :black : :white
axislegend(; position = (.82, .75), backgroundcolor = :transparent, color = text_color) # hide

fig, ax, zip_ob, Y_increments_geodesic, Y_increments_cayley # hide
end # hide

if Main.output_type == :html # hide
    save("retraction_comparison.png",        make_plot(; theme = :light)[1]; px_per_unit = 1.5) # hide
    save("retraction_comparison_dark.png",   make_plot(; theme = :dark )[1]; px_per_unit = 1.5) # hide
elseif Main.output_type == :latex # hide
    save("retraction_comparison.png",       make_plot(; theme = :light)[1]; px_per_unit = 2.0) # hide
end # hide

Main.include_graphics("retraction_comparison"; caption = raw"Comparison between the geodesic and the Cayley retraction.", width = .8) # hide
```

We see that for small ``\Delta`` increments the Cayley retraction seems to match the geodesic retraction very well, but for larger values there is a notable discrepancy:

```@setup s2_retraction
function plot_discrepancies(discrepancies; theme = :light)
    fig = Figure(; backgroundcolor = :transparent) # hide
    text_color = theme == :dark ? :white : :black # hide
    ax = Axis(fig[1, 1]; # hide
            backgroundcolor = :transparent, # hide
            xlabel = rich("η", font = :italic, color = text_color), # hide
            ylabel = rich("discrepancy", color = text_color), # hide
            ) # hide
    lines!(discrepancies; label = "Discrepancies between geodesic and Cayley retraction.", 
        linewidth = 2, color = mblue)

    axislegend(; position = (.22, .9), backgroundcolor = :transparent, color = text_color) # hide

    fig, ax
end
```

```@example s2_retraction
using LinearAlgebra: norm

_, __, zip_ob, Y_increments_geodesic, Y_increments_cayley = make_plot() # hide
discrepancies = [norm(Y_geo_inc - Y_cay_inc) for (Y_geo_inc, Y_cay_inc, _) in zip_ob]

if Main.output_type == :html # hide
    save("retraction_discrepancy.png",        plot_discrepancies(discrepancies; theme = :light)[1]; px_per_unit = 1.5) # hide
    save("retraction_discrepancy_dark.png",   plot_discrepancies(discrepancies; theme = :dark )[1]; px_per_unit = 1.5) # hide
elseif Main.output_type == :latex # hide
    save("retraction_discrepancy.png",        plot_discrepancies(discrepancies; theme = :light)[1]; px_per_unit = 2.0) # hide
end # hide

Main.include_graphics("retraction_discrepancy"; caption = raw"Discrepancy between the geodesic and the Cayley retraction.", width = .8) # hide
```

## In `GeometricMachineLearning`

The way we use *retractions*[^1] in `GeometricMachineLearning` is slightly different from their classical definition:

[^1]: Classical retractions are also defined in `GeometricMachineLearning` under the same name, i.e. there is e.g. a method [`cayley(::StiefelLieAlgHorMatrix)`](@ref) and a method [`cayley(::StiefelManifold, ::AbstractMatrix)`](@ref) (the latter being the classical retraction); but the user is *strongly discouraged* from using classical retractions as these are computational inefficient.

```@eval
Main.definition(raw"A **retraction** is a map ``\mathrm{Retraction}:\mathfrak{g}\mathrm{hor}\to\mathcal{M}`` such that 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Delta \mapsto \lambda(Y)\mathrm{Retraction}(\lambda(Y)^{-1}\Omega(\Delta)\lambda(Y))E,
" * Main.indentation * raw"```
" * Main.indentation * raw"is a classical retraction.")
```

We now discuss how two of these retractions, the geodesic retraction (exponential map) and the Cayley retraction, are implemented in `GeometricMachineLearning`.

## The Geodesic Retraction

The *geodesic retraction* is a retraction whose associated curve is also the unique geodesic. For many matrix Lie groups (including ``SO(N)``) geodesics are obtained by simply evaluating the exponential map [absil2008optimization, o1983semi](@cite):
 
```@eval
Main.theorem(raw"The geodesic on a matrix Lie group ``G`` with bi-invariant metric for ``B\in{}T_AG`` is simply
" * Main.indentation * raw"```math
" * Main.indentation * raw"\gamma(t) = \exp(t\cdotBA^-1)A,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\exp:\mathcal{g}\to{}G`` is the matrix exponential map.")
```

Starting from this basic map $\exp:\mathfrak{g}\to{}G$ we can build mappings for more complicated cases: 

1. **General tangent space to a Lie group** $T_AG$: The geodesic map for an element $V\in{}T_AG$ is simply $A\exp(A^{-1}V)$.

2. **Special tangent space to a homogeneous space** $T_E\mathcal{M}$: For $V=BE\in{}T_E\mathcal{M}$ the exponential map is simply $\exp(B)E$. 

3. **General tangent space to a homogeneous space** $T_Y\mathcal{M}$ with $Y = AE$: For $\Delta=ABE\in{}T_Y\mathcal{M}$ the exponential map is simply $A\exp(B)E$. This is the general case which we deal with.  

The general theory behind points 2. and 3. is discussed in chapter 11 of (O'Neill, 1983). The function `retraction` in `GeometricMachineLearning` performs $\mathfrak{g}^\mathrm{hor}\to\mathcal{M}$, which is the second of the above points. To get the third from the second point, we simply have to multiply with a matrix from the left. This step is done with `apply_section` and represented through the red vertical line in the diagram describing [general optimizer framework](@ref "Neural Network Optimizers").


### Word of caution

The Lie group corresponding to the Stiefel manifold $SO(N)$ has a bi-invariant Riemannian metric associated with it: $(B_1,B_2)\mapsto \mathrm{Tr}(B_1^TB_2)$.
For other Lie groups (e.g. the symplectic group) the situation is slightly more difficult (see (Bendokat et al, 2021).)

## Library Functions

```@docs; canonical=false
geodesic
cayley
```

## References 

```@bibliography
Pages = []
Canonical = false

absil2008optimization
bendokat2021real
o1983semi
```