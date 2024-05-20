# (Matrix) Manifolds

Manifolds are topological spaces that locally look like vector spaces. In the following we restrict ourselves to finite-dimensional smooth[^1] manifolds. In this section we routinely denote points on a manifold by lower case letters like ``x, y`` and ``z`` if we speak about general properties and by upper case letters like ``A`` and ``B`` if we talk about specific examples of matrix manifolds.

[^1]: *Smooth* here means ``C^\infty``.

```@eval 
Main.theorem(raw"A **finite-dimensional smooth manifold** of dimension ``n`` is a second-countable Hausdorff space ``\mathcal{M}`` for which ``\forall{}x\in\mathcal{M}`` we can find a neighborhood ``U`` that contains ``x`` and a corresponding homeomorphism ``\varphi_U:U\cong{}W\subset\mathbb{R}^n`` where ``W`` is an open subset. The homeomorphisms ``\varphi_U`` are referred to as *coordinate charts*. If two such coordinate charts overlap, i.e. if ``U_1\cap{}U_2\neq\{\}``, then the map ``\varphi_{U_2}^{-1}\circ\varphi_{U_1}`` is ``C^\infty``.")
```

One example of a manifold that is also important for `GeometricMachineLearning` is the Lie group[^2] of orthonormal matrices ``SO(N)``. Before we can proof that ``SO(N)`` is a manifold we first need the *preimage theorem*.

[^2]: Lie groups are manifolds that also have a *group structure*, i.e. there is an operation ``\mathcal{M}\times\mathcal{M}\to\mathcal{M},(a,b)\mapsto{}ab`` s.t. ``(ab)c = a(bc)`` and  there exists a neutral element``e\mathcal{M}`` s.t. ``ae`` = ``a`` ``\forall{}a\in\mathcal{M}`` as well as an (for every ``a``) inverse element ``a^{-1}`` s.t. ``a(a^{-1}) = e``. The neutral element ``e`` we refer to as ``\mathbb{I}`` when dealing with matrix manifolds.

## The Preimage Theorem

Before we can state the preimage theorem we need another definition: 

```@eval
Main.definition(raw"Consider a smooth mapping ``g: \mathcal{M}\to\mathcal{N}`` from one manifold to another. A point ``y\in\mathcal{N}`` is called a **regular value** of ``g`` if ``\forall{}x\in{}g^{-1}\{y\}`` the map ``T_xg:T_A\mathcal{M}\to{}T_{g(x)}\mathcal{N}`` is surjective.")
```

We now state the preimage theorem:

```@eval
Main.theorem(raw"Consider a smooth map ``g:\mathcal{M}\to\mathcal{N}`` from one manifold to another (we assume the dimensions of the two manifolds to be ``m+n`` and ``m``). Then the preimage of a regular point ``y`` of ``\mathcal{N}`` is a submanifold of ``\mathcal{M}``. Furthermore the codimension of ``g^{-1}\{y\}`` is equal to the dimension of ``\mathcal{N}`` and the tangent space ``T_x(g^{-1}\{y\})`` is equal to the kernel of ``T_xg``."; name = "Preimage Theorem")
```

__Proof__: Because ``\mathcal{N}`` has manifold structure we can find a chart ``\varphi_U:U\to\mathbb{R}^m`` for some neighborhood ``U`` that contains ``y``. We further consider a point ``A\in{}g^{-1}\{y\}`` and a chart around it ``\psi_V:V\to\mathbb{R}^{m+n}``. By the implicit function theorem we can then find a mapping ``h`` that turns ``\varphi_U\circ{}g\circ\psi_V^{-1}`` into a projection ``(x_1, \ldots, x_{n+m}) \mapsto (x_{n+1}, \ldots, x_{n+m})``. We now consider the neighborhood ``V_1\times\{0\}`` for ``V = V_1\times{}V_2`` with the coordinate chart ``(x_1, \ldots, x_n) \mapsto \psi(x_1, \ldots, x_n, 0, \ldots, 0).`` This proofs our assertion.


```@eval
Main.example(raw"The group ``SO(N)`` is a Lie group (i.e. has manifold structure).")
```

__Proof__: The vector space ``\mathbb{R}^{N\times{}N}`` clearly has manifold structure. The group ``SO(N)`` is equivalent to one of the level sets of the mapping: ``g:\mathbb{R}^{N\times{}N}\to\mathcal{S}(N), A\mapsto{}A^TA - \mathbb{I}``, i.e. it is the component of ``f^{-1}\{\mathbb{I}\}`` that contains ``\mathbb{I}``. We still need to proof that ``\mathbb{I}`` is a regular point of ``g``, i.e. that for ``A\in{}SO(N)`` the mapping ``T_Ag`` is surjective. This means that ``\forall{}B\in\mathcal{S}(N), A\in\mathbb{R}^{N\times{}N}`` ``\exists{}C\in\mathbb{R}^{N\times{}N}`` s.t. ``C^TA + A^TC = B``. The element ``C=\frac{1}{2}AB\in\mathcal{R}^{N\times{}N}`` satisfies this property.

Similarly we can also proof: 

```@eval
Main.example(raw"The sphere ``S^n:=\{x\in\mathbb{R}^{n+1}: x^Tx = 1\}`` is a manifold of dimension ``n``.")
```

__Proof__: Take ``g(x) = x^x - 1``.

## Tangent Spaces 

A tangent space can be seen as the *collection of all possible velocities a curve can take at a point on a manifold*. For this consider a manifold ``\mathcal{M}`` and a point ``x`` on it and the collection of ``C^\infty`` curves through ``x``: 

```@eval
Main.definition(raw"A mapping ``\gamma:(-\epsilon, \epsilon)\to\mathcal{M}`` that is ``C^\infty`` and for which we have ``\gamma(0) = x`` is called a **``C^\infty`` curve through ``x``**.")
```

The tangent space of ``\mathcal{M}`` at ``x`` is the collection of the first derivatives of all ``\gamma``: 

```@eval
Main.definition(raw"**The tangent space** of \mathcal{M} at ``x`` is the collection of all ``C^\infty`` curves at ``x`` modulo the equivalence class ``\gamma_1 \sim \gamma_2 \iff \gamma_1'(0) = \gamma_2'(0)``. It is denoted by ``T_x\mathcal{M}``.")
```

As is customary we write ``[\gamma]`` for the equivalence class of ``\gamma`` and this is by definition equivalent to ``\gamma'(0)``.
The tangent space ``T_x\mathcal{M}`` can be shown to be homeomorphic[^3] to ``\mathbb{R}^n`` where ``n`` is the dimension of the manifold ``\mathcal{M}``. If the homeomorphism is constructed through the coordinate chart ``(\varphi, U)`` we call it ``\varphi'(x)``. 

[^3]: Note that we have not formally defined addition for ``T_x\mathcal{M}``. This can be done through the definition ``[\gamma] + [\beta] = [\alpha]`` where ``\alpha`` is any ``C^\infty`` curve through ``x`` that satisfies ``\alpha'(0) = \beta(0) + \gamma(0)``. Note that we can always find such an ``\alpha`` by the [existence and uniqueness theorem](@ref "The Existence-And-Uniqueness Theorem").

We want to demonstrate this principle of constructing the tangent space from curves through the example of ``S^2``. We consider the following curves: 
1. ``\gamma_1(t) = \begin{pmatrix} 0 \\ sin(t) \\ cos(t) \end{pmatrix},``
2. ``\gamma_2(t) = \begin{pmatrix} sin(t) \\ 0 \\ cos(t) \end{pmatrix}.``
3. ``\gamma_3(t) = \begin{pmatrix} \exp(-t ^ 2 / 2)  t \sin(t) \\  \exp(-t ^ 2 / 2) t cos(t) \\  \sqrt{1 - (t ^ 2) exp(-t^2)} \end{pmatrix}. ``

We now plot the manifold ``S^2``, the three curves described above and the associated tangent vectors (visualized as arrows). Note that the tangent vectors induced by ``\gamma_1`` and ``\gamma_3`` are the same; for these curves we have ``\gamma_1 \sim \gamma_3`` and the tangent vectors of those two curves coincide: 

```@eval 
using CairoMakie
using ForwardDiff
using LaTeXStrings

function plot_curve!(p, gamma::Function; epsilon_range::T = 1.4, epsilon_spacing::T = .01, kwargs...) where T
    curve_domain = -epsilon_range : epsilon_spacing : epsilon_range
    curve = zeros(T, 3, length(curve_domain))
    for (i, t) in zip(axes(curve_domain, 1), curve_domain)
        curve[:, i] .= gamma(t)
    end
    lines!(p, curve[1, :], curve[2, :], curve[3, :]; kwargs...)
end

function plot_arrow!(p, gamma::Function; kwargs...) where T
    arrow_val = ForwardDiff.derivative(gamma, 0.)

    gamma_vec = ([gamma(0)[1]], [gamma(0)[2]], [gamma(0)[3]])
    gamma_deriv_vec = ([arrow_val[1]], [arrow_val[2]], [arrow_val[3]])

    arrows!(gamma_vec..., gamma_deriv_vec...; kwargs...)
end

function sphere(r, C)   # r: radius; C: center [cx,cy,cz]
    n = 100
    u = range(-π, π; length = n)
    v = range(0, π; length = n)
    x = C[1] .+ r * cos.(u) * sin.(v)'
    y = C[2] .+ r * sin.(u) * sin.(v)'
    z = C[3] .+ r * ones(n) * cos.(v)'
    x, y, z
end

function tangent_space(; n = 100)
    xs = LinRange(-1.2, 1.2, n)
    ys = LinRange(-1.2, 1.2, n)
    zs = [one(x) * one(y) for x in xs, y in ys]
    xs, ys, zs
end

fig, ax, plt = surface(sphere(1., [0., 0., 0.])...; alpha = .6)

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

for (i, curve, color) in zip(1:length(curves), curves, colors)
    plot_curve!(ax, curve; label = L"\gamma_%$(string(i))", linewidth = 2, color = color)
end

surface!(ax, tangent_space()...; alpha = .2)
text!(.9, -.9, 1.; text = L"T_x\mathcal{M}")

for (i, curve, color) in zip(1:length(curves), curves, colors)
    plot_arrow!(ax, curve; linewidth = .03, color = color)
end

axislegend(; position = (.75, .75))

if Main.output_type == :latex
    save("tangent_space.pdf", fig)
    elseif Main.output_type == :html
    save("tangent_space.svg", fig)
end

nothing
```

```@eval
using Markdown

if Main.output_type == :latex
    Markdown.parse(raw"""![Visualization of how the tangent space is constructed.]("manifolds/tangent_space.pdf")""")
    elseif Main.output_type == :html
    Markdown.parse(raw"""![]("tangent_space.svg")""")
end
```

The tangent space ``T_x\mathcal{M}`` (for ``x = \begin{pmatrix} 0 & 0 & 1\end{pmatrix}^T``) is also shown. 

## Vector Fields

A time-independent vector field[^4] is an object that specifies a velocity for every point on a domain. We first give the definition of a vector field on the vector space ``\mathbb{R}^n`` and limit ourselves here to ``C^\infty`` vector fields:

[^4]: Also called *ordinary differential equation* (ODE).

```@eval 
Main.definition(raw"A **vector field** on ``\mathbb{R}^n`` is a smooth map ``X:\mathbb{R}^n\to\mathbb{R}^n``.")
```

The definition of a vector field on a manifold is not much more complicated: 

```@eval 
Main.definition(raw"A **vector field** on ``\mathcal{M}`` is a map ``X`` defined on ``\mathcal{M}`` such that ``X(x)\in{}T_x\mathcal{M}`` and ``\varphi'\circ{}X\circ(\varphi)^{-1}`` is smooth for any coordinate chart ``(\varphi, U)`` that contains ``x``.")
```

In the section on the [existence-and-uniqueness theorem](@ref "The Existence-And-Uniqueness Theorem") we show that every vector field has a unique solution given an initial condition; i.e. given a point ``x\mathcal{M}`` and a vector field ``X`` we can find a curve ``\gamma`` such that ``\gamma(0) = x`` and ``\gamma'(t) = X(\gamma(t))`` for all ``t`` in some interval ``(-\epsilon, \epsilon)``.


## Library Functions 

```@docs; canonical=false
Manifold
```

## References 

```@bibliography
Pages = []
Canonical = false

absil2008optimization
```