# (Matrix) Manifolds

Manifolds are topological spaces that locally look like vector spaces. In the following we restrict ourselves to finite-dimensional smooth[^1] manifolds. In this section we routinely denote points on a manifold by lower case letters like ``x, y`` and ``z`` if we speak about general properties and by upper case letters like ``A`` and ``B`` if we talk about specific examples of matrix manifolds. All manifolds that can be used to build neural networks in `GeometricMachineLearning`, such as the [Stiefel manifold](@ref "The Stiefel Manifold") and the [Grassmann manifold](@ref "The Grassmann Manifold") are matrix manifolds.

[^1]: *Smooth* here means ``C^\infty``.

```@eval 
Main.definition(raw"A **finite-dimensional smooth manifold** of dimension ``n`` is a second-countable Hausdorff space ``\mathcal{M}`` for which ``\forall{}x\in\mathcal{M}`` we can find a neighborhood ``U``, that contains ``x,`` and a corresponding homeomorphism ``\varphi_U:U\cong{}W\subset\mathbb{R}^n`` where ``W`` is an open subset. The homeomorphisms ``\varphi_U`` are referred to as *coordinate charts*. If two such coordinate charts overlap, i.e. if ``U_1\cap{}U_2\neq\{\}``, then the map ``\varphi_{U_2}\circ\varphi_{U_1}^{-1}`` has to be ``C^\infty`` on ``varphi_{U_1}(U_1\cap{}U_2).`` We call the collection of coordinate charts ``\{\varphi_U\}_{U\subset\mathcal{M}}`` an **atlas** for ``\mathcal{M}.``")
```

One example of a manifold that is also important for `GeometricMachineLearning` is the Lie group[^2] of orthonormal matrices ``SO(N)``. Before we can proof that ``SO(N)`` is a manifold we first need the *preimage theorem*.

[^2]: Lie groups are manifolds that also have a *group structure*, i.e. there is an operation ``\mathcal{M}\times\mathcal{M}\to\mathcal{M},(a,b)\mapsto{}ab`` s.t. ``(ab)c = a(bc)`` and  there exists a neutral element``e\mathcal{M}`` s.t. ``ae`` = ``a`` ``\forall{}a\in\mathcal{M}`` as well as an (for every ``a``) inverse element ``a^{-1}`` s.t. ``a(a^{-1}) = e``. The neutral element ``e`` we refer to as ``\mathbb{I}`` when dealing with matrix manifolds.

## The Preimage Theorem

The preimage theorem is crucial for treating the specific manifolds that are part of `GeometricMachineLearning`; the preimage theorem gives spaces like the [Stiefel manifold](@ref "The Stiefel Manifold") the structure of a manifold. Before we can state the preimage theorem we need another definition:

```@eval
Main.definition(raw"Consider a smooth mapping ``g: \mathcal{M}\to\mathcal{N}`` from one manifold to another. A point ``y\in\mathcal{N}`` is called **regular point of ``g``** if ``\forall{}x\in{}g^{-1}\{y\}`` the map ``T_xg:T_x\mathcal{M}\to{}T_{y}\mathcal{N}`` is surjective.")
```

```@eval
Main.remark(raw"Here we already used the notation ``T_y\mathcal{N}`` to denote the *tangent space to ``\mathcal{N}`` at ``y``*. We will explain what we mean by this precisely below. For now we simply view ``T_y\mathcal{N}`` as *something that is homemorphic to ``\mathbb{R}^m``* and the *tangent map ``T_xg``* we will simply view as ``(\psi\circ{}g\circ{}\varphi^{-1})'(varphi(x)),`` where ``\varphi`` is a coordinate chart as ``x`` and ``\psi`` is a coordinate chart at ``y.`` In the examples we give below ``\mathcal{M}`` and ``\mathcal{N}`` will simply be vector spaces, and ``g`` will be differential map between vector spaces whose derivative at ``x\in{}f^{-1}\{y\}`` (for a regular point ``y``) is surjective. For a vector space ``\mathcal{V}`` we furthermore have ``T_x\mathcal{V} = \mathcal{V}.``")
```

We now state the preimage theorem:

```@eval
Main.theorem(raw"Consider a smooth map ``g:\mathcal{M}\to\mathcal{N}`` from one manifold to another (we assume the dimensions of the two manifolds to be ``m+n`` and ``m`` respectively). Then the preimage of a regular point ``y`` of ``\mathcal{N}`` is a submanifold of ``\mathcal{M}``. Furthermore the codimension of ``g^{-1}\{y\}`` is equal to the dimension of ``\mathcal{N}`` and the tangent space ``T_x(g^{-1}\{y\})`` is equal to the kernel of ``T_xg``."; name = "Preimage Theorem")
```

```@eval
Main.proof(raw"Because ``\mathcal{N}`` has manifold structure we can find a chart ``\psi_U:U\to\mathbb{R}^m`` for some neighborhood ``U`` that contains ``y``. We further consider a point ``x\in{}g^{-1}\{y\}`` and a chart around it ``\varphi_V:V\to\mathbb{R}^{m+n}``. By the implicit function theorem we can then find a mapping ``h`` that turns ``\psi_U\circ{}g\circ\varphi_V^{-1}`` into a projection ``(x_1, \ldots, x_{n+m}) \mapsto (x_{n+1}, \ldots, x_{n+m})``. We now consider the neighborhood ``V_1\times\{0\} = \varphi(V \cup f^{-1}\{y\})`` for ``\varphi(V) = V_1\times{}V_2`` with the coordinate chart ``(x_1, \ldots, x_n) \mapsto \varphi(x_1, \ldots, x_n, 0, \ldots, 0).`` As this map is also smooth by the implicit function theorem this proofs our assertion.")
```

We now give some examples of manifolds that can be constructed this way:

```@eval
Main.example(raw"The group ``SO(N)`` is a Lie group (i.e. has manifold structure). ``SO(N)`` has dimension ``N(N-1)/2.``")
```

```@eval
Main.proof(raw"The vector space ``\mathbb{R}^{N\times{}N}`` clearly has manifold structure. The group ``SO(N)`` is equivalent to one of the level sets of the mapping: ``g:\mathbb{R}^{N\times{}N}\to\mathcal{S}(N), A\mapsto{}A^TA - \mathbb{I}``, i.e. it is the component of ``f^{-1}\{\mathbb{I}\}`` that contains ``\mathbb{I}``; the image of ``f`` is contained in ``\mathcal{S}(N),`` the symmetric matrices of size ``N\times{}N.`` We still need to proof that ``\mathbb{I}`` is a regular point of ``g``, i.e. that for ``A\in{}SO(N)`` the mapping ``T_Ag`` is surjective. This means that ``\forall{}B\in\mathcal{S}(N)`` ``\exists{}C\in\mathbb{R}^{N\times{}N}`` s.t. ``C^TA + A^TC = B``. The element ``C=\frac{1}{2}AB\in\mathcal{R}^{N\times{}N}`` satisfies this property. The dimension of ``SO(N)`` is ``N(N-1)/2`` as ``\mathrm{dim}(\mathcal{S}(N))=N(N+1)/2.``")
```

Similarly we can also proof: 

```@eval
Main.example(raw"The sphere ``S^n:=\{x\in\mathbb{R}^{n+1}: x^Tx = 1\}`` is a manifold of dimension ``n``.")
```

```@eval
Main.proof(raw"Take ``g(x) = x^Tx - 1`` and proceed as in the case of ``SO(N)``.")
```

Note that both these manifolds, ``SO(N)`` and ``S^n`` are matrix manifolds, i.e. an element of ``\mathcal{M}`` can be written as an element of ``\mathbb{R}^{N\times{}N}`` in the first case and ``\mathbb{R}^{(n+1)\times{}1}`` in the second case. The additional conditions we impose on these manifolds are ``A^TA = \mathbb{I}`` in the first case and ``x^Tx = 1`` in the second case. Both of these manifolds belong to the category of [Stiefel manifolds](@ref "The Stiefel Manifold") and therefore also to the bigger category of matrix manifolds, i.e. every element of ``SO(N)`` and of ``S^n`` can be represented as a matrix on which further conditions are imposed (e.g orthogonality).

## The Immersion Theorem

The immersion theorem, similarly to the preimage theorem, gives another way of constructing a manifold based on a non-linear function. 

```@eval
Main.theorem(raw"Consider a differentiable function ``\mathcal{R}:\mathcal{N}\to\mathcal{M}`` with ``\mathrm{dim}(\mathcal{M}) = N > n = \mathrm{dim}(\mathcal{N})`` tangent mapping ``T_x\mathcal{R}`` has full rank at every point ``x\in\mathcal{N}``. Then ``\mathcal{R}(\mathcal{N})`` is a manifold *immersed* in ``\mathcal{M}``."; name = "Immersion Theorem")
```

The proof is again based on the [inverse function theorem](@ref "The Inverse Function Theorem").

```@eval
Main.proof(raw"Consider a point ``x\in\mathcal{N},`` a coordinate chart ``\varphi`` around ``x`` and a coordinate chart ``\psi`` around ``f(x).`` We now define the function
" * Main.indentation * raw"```math
" * Main.indentation * raw"    F:(x_1, \ldots, x_N) \mapsto (\psi\circ{}f\circ\varphi^{-1}(x_1, \ldots, x_n), x_{n+1}, \ldots, x_N).
" * Main.indentation * raw"```
" * Main.indentation * raw"By the inverse function theorem we can find an inverse of ``F`` for a neighborhood around the point ``(x_1, \ldots, x_n, 0, \ldots, 0)\in\mathbb{R}^N.`` We call this neighborhood ``V = V_1\times{}V_2`` and the inverse ``H.`` We now constrain ``V`` to the set ``V_1\times{}0``, which is isomorphic to a neighborhood around ``x`` in ``\mathbb{R}^n``. We then have in this neighborhood:
" * Main.indentation * raw"```math
" * Main.indentation * raw"    H(\psi\circ{}f\circ\varphi^{-1}(x_1, \ldots, x_n), 0, \ldots, 0) = (x_1, \ldots, x_n, 0, \ldots, 0), 
" * Main.indentation * raw"```
" * Main.indentation * raw"And we can take 
" * Main.indentation * raw"```math
" * Main.indentation * raw"    y \mapsto \pi\circ{}H(\psi(y), 0 \ldots, 0)
" * Main.indentation * raw"```
" * Main.indentation * raw"as our coordinate chart. ``\pi:\mathbb{R}^N\to\mathbb{R}^n`` is the projection onto the first ``n`` coordinates.")
```

We will use the immersion theorem when discussing the [symplectic solution manifold](@ref "The Symplectic Solution Manifold").

## Tangent Spaces 

We already alluded to tangent spaces when talking about the preimage and the immersion theorems. Here we will give a precise definition. A tangent space can be seen as the *collection of all possible velocities a curve can take at a point on a manifold*. For this consider a manifold ``\mathcal{M}`` and a point ``x`` on it and the collection of ``C^\infty`` curves through ``x``: 

```@eval
Main.definition(raw"A mapping ``\gamma:(-\epsilon, \epsilon)\to\mathcal{M}`` that is ``C^\infty`` and for which we have ``\gamma(0) = x`` is called a **``C^\infty`` curve through ``x``**.")
```

The tangent space of ``\mathcal{M}`` at ``x`` is the collection of the first derivatives of all ``\gamma``: 

```@eval
Main.definition(raw"The **tangent space** of ``\mathcal{M}`` at ``x`` is the collection of all ``C^\infty`` curves at ``x`` modulo the equivalence class ``\gamma_1 \sim \gamma_2 \iff \gamma_1'(0) = \gamma_2'(0)``. It is denoted by ``T_x\mathcal{M}``.")
```

As is customary we write ``[\gamma]`` for the equivalence class of ``\gamma`` and this is by definition equivalent to ``\gamma'(0)``.
The tangent space ``T_x\mathcal{M}`` can be shown to be homeomorphic[^3] to ``\mathbb{R}^n`` where ``n`` is the dimension of the manifold ``\mathcal{M}``. If the homeomorphism is constructed through the coordinate chart ``(\varphi, U)`` we call it ``\varphi'(x)`` or simply[^4] ``\varphi'``. If we are given a map ``g:\mathcal{M}\to\mathcal{N}`` we further define ``T_xg = (\varphi')^{-1}\circ(\varphi\circ{}g\circ\psi^{-1})'\circ{}\psi'``, i.e. a smooth map between two manifolds ``\mathcal{M}`` and ``\mathcal{N}`` induces a smooth map between the tangent spaces ``T_x\mathcal{M}`` and ``T_{g(x)}\mathcal{N}``.

[^3]: Note that we have not formally defined addition for ``T_x\mathcal{M}``. This can be done through the definition ``[\gamma] + [\beta] = [\alpha]`` where ``\alpha`` is any ``C^\infty`` curve through ``x`` that satisfies ``\alpha'(0) = \beta(0) + \gamma(0)``. Note that we can always find such an ``\alpha`` by the [existence and uniqueness theorem](@ref "The Existence-And-Uniqueness Theorem").

[^4]: We will further discuss this when we introduce the [tangent bundle](@ref "The Tangent Bundle").

We want to demonstrate this principle of constructing the tangent space from curves through the example of ``S^2``. We consider the following curves: 
1. ``\gamma_1(t) = \begin{pmatrix} 0 \\ \sin(t) \\ \cos(t) \end{pmatrix},``
2. ``\gamma_2(t) = \begin{pmatrix} \sin(t) \\ 0 \\ \cos(t) \end{pmatrix},``
3. ``\gamma_3(t) = \begin{pmatrix} \exp(-t ^ 2 / 2)  t \sin(t) \\  \exp(-t ^ 2 / 2) t \cos(t) \\  \sqrt{1 - (t ^ 2) \exp(-t^2)} \end{pmatrix}. ``

We now plot the manifold ``S^2``, the three curves described above and the associated tangent vectors (visualized as arrows). Note that the tangent vectors induced by ``\gamma_1`` and ``\gamma_3`` are the same; for these curves we have ``\gamma_1 \sim \gamma_3`` and the tangent vectors of those two curves coincide: 

```@eval 
using CairoMakie
using ForwardDiff
using LaTeXStrings

function plot_curve!(ax, gamma::Function; epsilon_range::T = 1.4, epsilon_spacing::T = .01, kwargs...) where T
    curve_domain = -epsilon_range : epsilon_spacing : epsilon_range
    curve = zeros(T, 3, length(curve_domain))
    for (i, t) in zip(axes(curve_domain, 1), curve_domain)
        curve[:, i] .= gamma(t)
    end
    lines!(ax, curve[1, :], curve[2, :], curve[3, :]; kwargs...)
end

function plot_arrow!(ax, gamma::Function; kwargs...)
    arrow_val = ForwardDiff.derivative(gamma, 0.)

    gamma_vec = ([gamma(0)[1]], [gamma(0)[2]], [gamma(0)[3]])
    gamma_deriv_vec = ([arrow_val[1]], [arrow_val[2]], [arrow_val[3]])

    arrows!(ax, gamma_vec..., gamma_deriv_vec...; kwargs...)
end

function tangent_space(; n = 100)
    xs = LinRange(-1.2, 1.2, n)
    ys = LinRange(-1.2, 1.2, n)
    zs = [one(x) * one(y) for x in xs, y in ys]
    xs, ys, zs
end

gamma_1(t) = [zero(t), sin(t), cos(t)]
gamma_2(t) = [sin(t), zero(t), cos(t)]
gamma_3(t) = [exp(-t ^ 2 / 2) * (t ^ 1) * sin(t), exp(-t ^ 2 / 2) * (t ^ 1) * cos(t), sqrt(1 - (t ^ 2) * exp(-t^2))]

curves = (gamma_1, gamma_2, gamma_3)

morange = RGBf(255 / 256, 127 / 256, 14 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)

colors = (morange, mblue, mred)

function make_plot(; theme = :light)
    text_color = theme == :light ? :black : :white

    fig = Figure(; backgroundcolor = :transparent)

    ax = Axis3(fig[1, 1]; 
        backgroundcolor = :transparent, 
        aspect = (1., 1., 0.8), 
        azimuth = π / 6, 
        elevation = π / 8,
        xlabel = L"x_1",
        ylabel = L"x_2",
        zlabel = L"x_3",
        xlabelcolor = text_color,
        ylabelcolor = text_color,
        zlabelcolor = text_color,
        )

    surface!(Main.sphere(1., [0., 0., 0.])...; alpha = .6)

    for (i, curve, color) in zip(1:length(curves), curves, colors)
        plot_curve!(ax, curve; label = rich("γ", subscript(string(i)); color = text_color, font = :italic), linewidth = 2, color = color)
    end

    surface!(ax, tangent_space()...; alpha = .2)
    text!(.9, -.9, 1.; text = L"T_x\mathcal{M}", color = text_color)

    for (i, curve, color) in zip(1:length(curves), curves, colors)
        plot_arrow!(ax, curve; linewidth = .03, color = color)
    end

    axislegend(; position = (.82, .75), backgroundcolor = :transparent, color = text_color)

    fig, ax
end

if Main.output_type == :html
    save("tangent_space.png",        make_plot(; theme = :light)[1]; px_per_unit = 1.5)
    save("tangent_space_dark.png",   make_plot(; theme = :dark )[1]; px_per_unit = 1.5)
elseif Main.output_type == :latex
    save("tangent_space.png",       make_plot(; theme = :light)[1]; px_per_unit = 2.0)
end

nothing
```

```@example
Main.include_graphics("tangent_space"; caption = raw"Visualization of how the tangent space is constructed. ", width = .9) # hide
```

The tangent space ``T_x\mathcal{M}`` for

```math
x = \begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}
```

 is also shown. 

## Vector Fields

A time-independent vector field[^5] is an object that specifies a velocity for every point on a domain. We first give the definition of a vector field on the vector space ``\mathbb{R}^n`` and limit ourselves here to ``C^\infty`` vector fields:

[^5]: Also called *ordinary differential equation* (ODE).

```@eval 
Main.definition(raw"A **vector field** on ``\mathbb{R}^n`` is a smooth map ``X:\mathbb{R}^n\to\mathbb{R}^n``.")
```

The definition of a vector field on a manifold is not much more complicated: 

```@eval 
Main.definition(raw"A **vector field** on ``\mathcal{M}`` is a map ``X`` defined on ``\mathcal{M}`` such that ``X(x)\in{}T_x\mathcal{M}`` and ``\varphi'\circ{}X\circ(\varphi)^{-1}`` is smooth for any coordinate chart ``(\varphi, U)`` that contains ``x``.")
```

In the section on the [existence-and-uniqueness theorem](@ref "The Existence-And-Uniqueness Theorem") we show that every vector field has a unique solution given an initial condition; i.e. given a point ``x\in\mathcal{M}`` and a vector field ``X`` we can find a curve ``\gamma`` such that ``\gamma(0) = x`` and ``\gamma'(t) = X(\gamma(t))`` for all ``t`` in some interval ``(-\epsilon, \epsilon)``.


## The Tangent Bundle

To each manifold ``\mathcal{M}`` we can associate another manifold which we call the *tangent bundle* and denote by ``T\mathcal{M}``. The points on this manifold are: 

```math
T\mathcal{M} = \{ (x, v_x): x\in\mathcal{M},\, v_x\in{}T_x\mathcal{M} \}.
```

Coordinate charts on this manifold can be constructed in a straightforward manner; for every coordinate chart ``\varphi_U`` the map ``\varphi_U'(x)`` gives a homeomorphism between ``T_x\mathcal{M}`` and ``\mathbb{R}^n`` for any ``x\in{}U``. We can then find a neighborhood of any point ``(x, v_x)`` by taking ``\pi^{-1}(U) = \{(x, v_x): x\in{}U, v_x\in{}T_x\mathcal{M}\}`` and this neighborhood is isomorphic to ``\mathbb{R}^{2n}`` via ``(x, v_x) \mapsto (\varphi_U(x), \varphi'(x)v_x)``. The [geodesic spray](@ref "Geodesic Sprays and the Exponential Map") is an important vector field defined on ``T\mathcal{M}``.

## Library Functions 

```@docs
Manifold
rand(::Type{MT}, ::Integer, ::Integer) where MT <: Manifold
rand(::GeometricMachineLearning.Backend, ::Type{MT}, ::Integer, ::Integer) where MT <: Manifold
```

## References 

```@bibliography
Pages = []
Canonical = false

absil2008optimization
```