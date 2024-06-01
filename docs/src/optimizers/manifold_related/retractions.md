# Retractions

In practice we usually do not solve the geodesic equation exactly in each optimization step (even though this is possible and computationally feasible), but prefer approximations that are called "retractions" [absil2008optimization](@cite) for stability. The definition of a retraction in `GeometricMachineLearning` is slightly different from how it is usually defined in textbooks [absil2008optimization, hairer2006geometric](@cite). We discuss these differences here.

## Classical Retractions

By "classical retraction" we here mean the textbook definition. 

```@eval
Main.theorem(raw"A **classical retraction** is a smooth map
" * Main.indentation * raw"```math 
" * Main.indentation * raw"R: T\mathcal{M}\to\mathcal{M}:(x,v)\mapsto{}R_x(v),
" * Main.indentation * raw"```
" * Main.indentation * raw"such that each curve ``c(t) := R_x(tv)`` is a local approximation of a geodesic, i.e. the following two conditions hold:
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

We should mention that the factor ``\frac{1}{2}`` is sometimes left out in the definition of the Cayley transform when used in different contexts. But it is necessary for defining a retraction. 

We want to compare the [`geodesic`](@ref) retraction with the [`cayley`](@ref) retraction for the example we already introduced when talking about the [exponential map](@ref "Geodesic Sprays and the Exponential Map"):

```@setup s2_retraction
using GLMakie

include("../../../gl_makie_transparent_background_hack.jl")
```

```@setup s2_retraction
using GeometricMachineLearning
import Random # hide
Random.seed!(123) # hide

Y = rand(StiefelManifold, 3, 1)

v = 5 * rand(3, 1)
Δ = v - Y * (v' * Y)

function do_setup(; theme=:light)
    text_color = theme == :dark ? :white : :black # hide
    fig = Figure(; backgroundcolor = :transparent) # hide
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
η_increments = 0.1 : 0.1 : 5.5
Δ_increments = [Δ * η for η in η_increments]

Y_increments_geodesic = [geodesic(Y, Δ_increment) for Δ_increment in Δ_increments]
Y_increments_cayley = [cayley(Y, Δ_increment) for Δ_increment in Δ_increments]
 # hide
function make_plot(; theme=:light) # hide

text_color = theme == :light ? :black : :white # hide

fig, ax, point_vec = do_setup(; theme = theme) # hide

Y_zeros = zeros(length(Y_increments_geodesic))
Y_geodesic_reshaped = [copy(Y_zeros), copy(Y_zeros), copy(Y_zeros)]
Y_cayley_reshaped = [copy(Y_zeros), copy(Y_zeros), copy(Y_zeros)]

zip_ob = zip(Y_increments_geodesic, Y_increments_cayley, axes(Y_increments_geodesic, 1))

for (Y_increment_geodesic, Y_increment_cayley, i) in zip_ob
    for d in (1, 2, 3)
        Y_geodesic_reshaped[d][i] = Y_increment_geodesic[d]

        Y_cayley_reshaped[d][i] = Y_increment_cayley[d]
    end
end

scatter!(ax, Y_geodesic_reshaped...; 
        color = mred, markersize = 5, label = rich("geodesic retraction"; color = text_color))

scatter!(ax, Y_cayley_reshaped...; 
        color = mblue, markersize = 5, label = rich("Cayley retraction"; color = text_color))

arrow_vec = ([Δ[1]], [Δ[2]], [Δ[3]]) # hide
arrows!(ax, point_vec..., arrow_vec...; color = mred, linewidth = .02) # hide
axislegend(; position = (.82, .75), backgroundcolor = :transparent, color = text_color) # hide

fig, ax, zip_ob, Y_increments_geodesic, Y_increments_cayley # hide
end # hide

fig_light = make_plot(; theme = :light)[1] # hide
fig_dark = make_plot(; theme = :dark)[1] # hide
save("retraction_comparison.png",        fig_light |> alpha_colorbuffer) #; px_per_unit = 1.5) # hide
save("retraction_comparison_dark.png",   fig_dark |> alpha_colorbuffer) #; px_per_unit = 1.5) # hide

Main.include_graphics("retraction_comparison"; caption = raw"Comparison between the geodesic and the Cayley retraction.", width = .8) # hide
```

We see that for small ``\Delta`` increments the Cayley retraction seems to match the geodesic retraction very well, but for larger values there is a notable discrepancy:

```@setup s2_retraction
using CairoMakie
function plot_discrepancies(discrepancies; theme = :light)
    fig = Figure(; backgroundcolor = :transparent) # hide
    text_color = theme == :dark ? :white : :black # hide
    ax = Axis(fig[1, 1]; # hide
            backgroundcolor = :transparent, # hide
            xlabel = rich("η", font = :italic, color = text_color), # hide
            ylabel = rich("discrepancy", color = text_color), # hide
            ) # hide
    lines!(η_increments, discrepancies; label = rich("Discrepancies between geodesic and Cayley retraction", color = text_color), 
        linewidth = 2, color = mblue)

    axislegend(; position = (.22, .9), backgroundcolor = :transparent, color = text_color) # hide

    fig, ax
end
```

```@example s2_retraction
using LinearAlgebra: norm

_, __, zip_ob, Y_increments_geodesic, Y_increments_cayley = make_plot() # hide
discrepancies = [norm(Y_geo_inc - Y_cay_inc) for (Y_geo_inc, Y_cay_inc, _) in zip_ob]

fig_light = plot_discrepancies(discrepancies; theme = :light)[1] # hide
fig_dark = plot_discrepancies(discrepancies; theme = :dark)[1] # hide
save("retraction_discrepancy.png",        fig_light) #; px_per_unit = 1.5) # hide
save("retraction_discrepancy_dark.png",   fig_dark) #; px_per_unit = 1.5) # hide

Main.include_graphics("retraction_discrepancy"; caption = raw"Discrepancy between the geodesic and the Cayley retraction.", width = .6) # hide
```

## In `GeometricMachineLearning`

The way we use *retractions*[^1] in `GeometricMachineLearning` is slightly different from their classical definition:

[^1]: Classical retractions are also defined in `GeometricMachineLearning` under the same name, i.e. there is e.g. a method [`cayley(::StiefelLieAlgHorMatrix)`](@ref) and a method [`cayley(::StiefelManifold, ::AbstractMatrix)`](@ref) (the latter being the classical retraction); but the user is *strongly discouraged* from using classical retractions as these are computational inefficient.

```@eval
Main.definition(raw"Given a section ``\lambda:\mathcal{M}\to{}G`` a **retraction** is a map ``\mathrm{Retraction}:\mathfrak{g}^\mathrm{hor}\to{}G`` such that 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Delta \mapsto \lambda(Y)\mathrm{Retraction}(\lambda(Y)^{-1}\Omega(\Delta)\lambda(Y))E,
" * Main.indentation * raw"```
" * Main.indentation * raw"is a classical retraction.")
```

We now discuss how two of these retractions, the geodesic retraction (exponential map) and the Cayley retraction, are implemented in `GeometricMachineLearning`.

## Retractions for Homogeneous Spaces

Here we harness special properties of homogeneous spaces to obtain computationally efficient retractions for the [Stiefel manifold](@ref "The Stiefel Manifold") and the [Grassmann manifold](@ref "The Grassmann Manifold"). This is also discussed in e.g. [bendokat2020grassmann, bendokat2021real](@cite).

The *geodesic retraction* is a retraction whose associated curve is also the unique geodesic. For many matrix Lie groups (including ``SO(N)``) geodesics are obtained by simply evaluating the exponential map [absil2008optimization, o1983semi](@cite):
 
```@eval
Main.theorem(raw"The geodesic on a compact matrix Lie group ``G`` with bi-invariant metric for ``B\in{}T_AG`` is simply
" * Main.indentation * raw"```math
" * Main.indentation * raw"\gamma(t) = \exp(t\cdot{}BA^{-1})A,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\exp:\mathcal{g}\to{}G`` is the matrix exponential map.")
```

Because ``SO(N)`` is compact and we furnish it with the canonical metric, i.e. 

```math
    g:T_AG\times{}T_AG \to \mathbb{R}, (B_1, B_2) \mapsto \mathrm{Tr}(B_1^TB_2) = \mathrm{Tr}((B_1A^{-1})^T(B_2A^{-1})),
```

its geodesics are thus equivalent to the exponential maps. We now use this observation to obtain expression for the geodesics on the [Stiefel manifold](@ref "The Stiefel Manifold") ``St(n, N)``. We use the following theorem from [o1983semi; Proposition 25.7](@cite):

```@eval
Main.theorem(raw"The geodesics for a naturally-reductive homogeneous space ``\mathcal{M}`` starting at ``Y`` are given by:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\gamma_{\Delta}(t) = \exp(t\cdot\Omega(\Delta))Y,
" * Main.indentation * raw"```
" * Main.indentation * raw"where the ``\exp`` is the exponential map for the Lie group ``G`` corresponding to ``\mathcal{M}``.")
```

The theorem requires the homogeneous space to be naturally reductive: 

```@eval
Main.definition(raw"A homogeneous space is called **naturally-reductive** if the following two conditions hold:
" * Main.indentation * raw"1. ``A^{-1}BA\in\mathfrak{g}^\mathrm{hor}`` for every ``B\in\mathfrak{g}^\mathrm{hor}`` and ``A\in\exp(\mathfrak{g}^\mathrm{ver}``),
" * Main.indentation * raw"2. ``g([X, Y]^\mathrm{hor}, Z) = g(X, [Y, Z]^\mathrm{hor})`` for all ``X, Y, Z \in \mathfrak{g}^\mathrm{hor}``,
" * Main.indentation * raw"where ``[X, Y]^\mathrm{hor} = \Omega(XYE - YXE)``. If only the first condition holds the homogeneous space is called **reductive** but not **naturally-reductive**.")
```

We state here without proof that the [Stiefel manifold](@ref "The Stiefel Manifold") and the [Grassmann manifold](@ref "The Grassmann Manifold") are naturally-reductive. An empirical verification of this is very easy:

```@example naturally_reductive
using GeometricMachineLearning
import Random # hide
Random.seed!(123) # hide

B = rand(SkewSymMatrix, 6) # ∈ 𝔤
A = exp(B - StiefelLieAlgHorMatrix(B, 3)) # ∈ exp(𝔤ᵛᵉʳ)

X = rand(StiefelLieAlgHorMatrix, 6, 3) # ∈ 𝔤ʰᵒʳ
Y = rand(StiefelLieAlgHorMatrix, 6, 3) # ∈ 𝔤ʰᵒʳ
Z = rand(StiefelLieAlgHorMatrix, 6, 3) # ∈ 𝔤ʰᵒʳ

@assert StiefelLieAlgHorMatrix(A' * X * A, 3) ≈ A' * X * A # hide
A' * X * A # this has to be in 𝔤ʰᵒʳ for St(3, 6) to be reductive
```

verifies the first property and

```@example naturally_reductive
using LinearAlgebra: tr
adʰᵒʳ(X, Y) = StiefelLieAlgHorMatrix(X * Y - Y * X, 3)

@assert tr(adʰᵒʳ(X, Y)' * Z) ≈ tr(X' * adʰᵒʳ(Y, Z)) # hide
tr(adʰᵒʳ(X, Y)' * Z) ≈ tr(X' * adʰᵒʳ(Y, Z))
```

verifies the second.

In `GeometricMachineLearning` we always work with elements in ``\mathfrak{g}^\mathrm{hor}`` and the Lie group ``G`` is always ``SO(N)``. We hence use:

```math
    \gamma_\Delta(t) = \exp(\lambda(Y)\lambda(Y)^{-1}\Omega(\Delta)\lambda(Y)\lambda(Y)^{-1})Y = \lambda(Y)\exp(\lambda(Y)^{-1}\Omega(\Delta)\lambda(Y))E,
```

where we have used that 

```math
 \exp(\Lambda{}B\Lambda^{-1}) = \sum_{n = 0}^\infty \frac{1}{n!}(\Lambda{}B\Lambda^{-1})^n = \sum_{n = 0}^\infty \frac{1}{n!}\underbrace{(\Lambda{}B\Lambda^{-1})}_{\text{$n$ times}} = \sum_{n = 0}^\infty \Lambda\frac{1}{n!}B^n\Lambda^{-1}.
```

Based on this we define the maps: 

```math
\mathtt{geodesic}: \mathfrak{g}^\mathrm{hor} \to G, B \mapsto \exp(B),
```

and

```math
\mathtt{cayley}: \mathfrak{g}^\mathrm{hor} \to G, B \mapsto \mathrm{Cayley}(B),
```

where ``B = \lambda(Y)^{-1}\Omega(\Delta)\lambda(Y)``. These expressions for `geodesic` and `cayley` are the ones that we typically use in `GeometricMachineLearning` for computational reasons. We show how we can utilize the sparse structure of ``\mathfrak{g}^\mathrm{hor}`` for computing the geodesic retraction and the Cayley retraction (i.e. the expressions ``\exp(B)`` and ``\mathrm{Cayley}(B)`` for ``B\in\mathfrak{g}^\mathrm{hor}``). Similar derivations can be found in [celledoni2000approximating, fraikin2007optimization, bendokat2021real](@cite).

```@eval
Main.remark(raw"Further note that, even though the global section ``\lambda:\mathcal{M} \to G`` is not unique, the final geodesic ``
\gamma_\Delta(t) = \lambda(Y)\exp(\lambda(Y)^{-1}\Omega(\Delta)\lambda(Y))E`` does not depend on the particular section we choose.")
```

### The Geodesic Retraction

An element of ``\mathfrak{g}^\mathrm{hor}`` can be written as:

```math
\begin{bmatrix}
    A & -B^T \\ 
    B & \mathbb{O}
\end{bmatrix} = \begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix} =: B'(B'')^T,
```

where we exploit the sparse structure of the array, i.e. it is a multiplication of a ``N\times2n`` with a ``2n\times{}N`` matrix.

We further use the following: 

```math
    \begin{aligned}
    \exp(B'(B'')^T) & = \sum_{n=0}^\infty \frac{1}{n!} (B'(B'')^T)^n = \mathbb{I} + \sum_{n=0}^\infty \frac{1}{n!} B'\sum_{n=1}^\infty B'((B'')^TB')(B'')^T \\
    & = \mathbb{I} + B'\left( \sum_{n=1}^\infty \frac{1}{n!} ((B'')^TB')^{n-1} \right)B'' =: \mathbb{I} + B'\mathfrak{A}(B', B'')B'',
    \end{aligned}
```

where we defined ``\mathfrak{A}(B', B'') := \sum_{n=1}^\infty \frac{1}{n!} ((B'')^TB')^{n-1}.`` Note that evaluating ``\mathfrak{A}`` relies on computing products of *small* matrices of size ``2n\times2n.`` We do this by relying on a simple Taylor expansion (see the docstring for [`GeometricMachineLearning.𝔄`](@ref)). 

The final expression we obtain is: 

```math
\exp(B) = \mathbb{I} + B' \mathfrak{A}(B', B'')  (B'')^T
```

### The Cayley Retraction

For the Cayley retraction we leverage the decomposition of ``B = B'(B'')^T\in\mathfrak{g}^\mathrm{hor}`` through the *Sherman-Morrison-Woodbury formula*:

```math
(\mathbb{I} - \frac{1}{2}B'(B'')^T)^{-1} = \mathbb{I} + \frac{1}{2}B'(\mathbb{I} - \frac{1}{2}B'(B'')^T)^{-1}(B'')^T
```

So what we have to compute the inverse of:

```math
\mathbb{I} - \frac{1}{2}\begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix}\begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} = 
\begin{bmatrix}  \mathbb{I} - \frac{1}{4}A & - \frac{1}{2}\mathbb{I} \\ \frac{1}{2}B^TB - \frac{1}{8}A^2 & \mathbb{I} - \frac{1}{4}A  \end{bmatrix}.
```

By leveraging the sparse structure of the matrices in ``\mathfrak{g}^\mathrm{hor}`` we arrive at the following expression for the Cayley retraction (similar to the case of the geodesic retraction):

```math
\mathrm{Cayley}(B) = \mathbb{I} + \frac{1}{2} B' (\mathbb{I}_{2n} - \frac{1}{2} (B'')^T B')^{-1} (B'')^T (\mathbb{I} + \frac{1}{2} B),
```

where we have abbreviated ``\mathbb{I} := \mathbb{I}_N.`` We conclude with a remark:

```@eval
Main.remark(raw"As mentioned previously the Lie group ``SO(N)``, i.e. the one corresponding to the Stiefel manifold and the Grassmann manifold, has a bi-invariant Riemannian metric associated with it: ``(B_1,B_2)\mapsto \mathrm{Tr}(B_1^TB_2)``. For other Lie groups (e.g. the symplectic group) the situation is slightly more difficult [bendokat2021real](@cite).")
```

## Library Functions

```@docs; canonical=false
geodesic(::StiefelLieAlgHorMatrix)
geodesic(::GrassmannLieAlgHorMatrix)
cayley(::StiefelLieAlgHorMatrix)
cayley(::GrassmannLieAlgHorMatrix)
cayley(::Manifold{T}, ::AbstractMatrix{T}) where T
GeometricMachineLearning.𝔄
```

## References 

```@bibliography
Pages = []
Canonical = false

absil2008optimization
bendokat2021real
o1983semi
```