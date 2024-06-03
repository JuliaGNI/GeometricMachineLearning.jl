# (Topological) Metric Spaces 

A metric space is a certain class of a topological space where the topology is *induced through a metric*.

```@eval
Main.definition(raw"A **metric** on a topological space ``\mathcal{M}`` is a mapping ``d:\mathcal{M}\times\mathcal{M}\to\mathbb{R}`` such that the following three conditions hold: 
" * 
Main.indentation * raw"1. ``d(x, y) = 0 \iff x = y`` for every ``x,y\in\mathcal{M}``, i.e. the distance between 2 points is only zero if and only if they are the same,
" * 
Main.indentation * raw"2. ``d(x, y) = d(y, x)``,
" *
Main.indentation * raw"3. ``d(x, z) \leq d(x, y) + d(y, z)``.
" *
Main.indentation * raw"The second condition is referred to as *symmetry* and the third condition is referred to as the *triangle inequality*.")
```

We give some examples of metric spaces that are relevant for us: 

```@eval
Main.example(raw"The real line ``\mathbb{R}`` with the metric defined by the absolute distance between two points: ``d(x, y) = |y - x|``.")
```

```@eval
Main.example(raw"The vector space ``\mathbb{R}^n`` with the *Euclidean distance* ``d_2(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}``.")
```

```@eval
Main.example(raw"The space of continuous functions ``\mathcal{C} = \{f:(-\epsilon, \epsilon)\to\mathbb{R}^n\}`` with the metric ``d_\infty(f_1, f_2) = \mathrm{sup}_{t\in(-\epsilon, \epsilon)}|f_1(t) - f_2(t)|.``")
```

```@eval
Main.proof(raw"We have to show the triangle inequality: 
" * 
Main.indentation * raw"```math
" * 
Main.indentation * raw"\begin{aligned}
" *
Main.indentation * raw"d_\infty(d_1, d_3) = \mathrm{sup}_{t\in(-\epsilon, \epsilon)}|f_1(t) - f_3(t)| & \leq \mathrm{sup}_{t\in(-\epsilon, \epsilon)}(|f_1(t) - f_2(t)| + |f_2(t) - f_3(t)|) \\
" *
Main.indentation * raw"& \leq \mathrm{sup}_{t\in(-\epsilon, \epsilon)}|f_1(t) - f_2(t)| + \mathrm{sup}_{t\in(-\epsilon, \epsilon)}|f_1(t) - f_2(t)|.
" * 
Main.indentation * raw"\end{aligned}
" * 
Main.indentation * raw"```
" *
Main.indentation * raw"This shows that ``d_\infty`` is indeed a metric.")
```

```@eval
Main.example(raw"Any Riemannian manifold is a metric space.")
```

This last example shows that *metric spaces need not be vector spaces*, i.e. spaces for which we can define a metric but not addition of two elements. This will be discussed in more detail in the section on [riemannian manifolds](@ref "Riemannian Manifolds").

## Complete Metric Spaces

To define *complete metric spaces* we first need the definition of a *Cauchy sequence*.

```@eval
Main.definition(raw"A **Cauchy sequence** is a sequence ``(a_n)_{n\in\mathbb{N}}`` for which, given any `epsilon>0`, we can find an integer ``N`` such that ``d(a_n, a_m) < \epsilon`` for all ``n, m \geq N``.")
```

Now we can give the definition of a *complete metric space*:

```@eval
Main.definition(raw"A **complete metric space** is one for which every Cauchy sequence converges.")
```

Completeness of the real numbers is most often seen as an axiom and therefore stated without proof. This also implies completeness of ``\mathbb{R}^n`` [lang2012real](@cite).


# (Topological) Vector Spaces

Vector Spaces are, like metric spaces, topological spaces which we endow with additional structure. 

```@eval
Main.definition(raw"A **vector space** ``\mathcal{V}`` is a topological space for which we define an operation called *addition* and denoted by ``+`` and an operation called *scalar multiplication* (by elements of ``\mathbb{R}``) denoted by ``x \mapsto ax`` for ``x\in\mathcal{V}`` and ``x\in\mathbb{R}`` for which the following hold for all ``x, y, z\in\mathcal{V}`` and ``a, b\in\mathbb{R}``:
" * 
Main.indentation * raw"1. ``x + (y + z) = (x + y) + z,``
" * 
Main.indentation * raw"2. ``x + y = y + x,``
" * 
Main.indentation * raw"3. ``\exists 0 \in \mathcal{V}`` such that ``x + 0 = x,``
" * 
Main.indentation * raw"4. ``\exists -x \in \mathcal{V} such that ``x + (-x) = 0,``
" * 
Main.indentation * raw"5. ``a(ax) = (ab)x,``
" * 
Main.indentation * raw"6. ``1x = x`` for ``1\in\mathbb{R},``
" * 
Main.indentation * raw"7. ``a(x + y) = ax + ay,``
" * 
Main.indentation * raw"8. ``(a + b)x = ax + bx.``
" * 
Main.indentation * raw"The first law is known as *associativity*, the second one as *commutativity* and the last two ones are known *distributivity*.")
```

The topological spaces ``\mathbb{R}`` and ``\mathbb{R}^{n}`` are (almost) trivially vector spaces. The same is true for many function spaces. One of the special aspects of `GeometricMachineLearning` is that it can deal with spaces that are not vector spaces, but manifolds. All vector spaces are however manifolds.  

```@bibliography
Pages = []
Canonical = false

lang2012real
```