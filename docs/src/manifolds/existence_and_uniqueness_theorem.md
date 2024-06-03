# The Existence-And-Uniqueness Theorem

The *existence-and-uniqueness theorem*, also known as the *Picard-Lindelöf theorem*, *Picard's existence theorem* and the *Cauchy-Lipschitz theorem* gives a proof of the existence of solutions for ODEs. Here we state the existence-and-uniqueness theorem for manifolds as vector fields are just a special case of this. Its proof relies on the [Banach fixed-point theorem](@ref "The Fixed-Point Theorem")[^1].

[^1]: It has to be noted that the proof given here is not entirely self-contained. The proof of the fundamental theorem of calculus, i.e. the proof of the existence of an antiderivative of a continuous function [lang2012real](@cite), is omitted for example. 

```@eval
Main.theorem(raw"Let ``X`` a vector field on the manifold ``\mathcal{M}`` that is differentiable at ``x``. Then we can find an ``\epsilon>0`` and a unique curve ``\gamma:(-\epsilon, \epsilon)\to\mathcal{M}`` such that ``\gamma'(t) = X(\gamma(t))``."; name = "Existence-And-Uniqueness Theorem")
```

```@eval
Main.proof(raw"We consider a ball around a point ``x\in\mathcal{M}`` with radius ``r`` that we pick such that the ball ``B(x, r)`` fits into the ``U`` of some coordinate chart ``\varphi_U``; we further use ``X`` and ``\varphi'\circ{}X\circ\varphi^{-1}`` interchangeably in this proof. We then define ``L := \mathrm{sup}_{y,z\in{}B(x,r)}|X(y) - X(z)|/|y - z|.`` Note that this ``L`` is always finite because ``X`` is bounded and differentiable. We now define the map ``\Gamma: C^\infty((-\epsilon, \epsilon), \mathbb{R}^n)\to{}C^\infty((-\epsilon, \epsilon), \mathbb{R}^n)`` (for some ``\epsilon`` that we do not yet fix) as 
" * 
Main.indentation * raw"```math
" * 
Main.indentation * raw"\Gamma\gamma(t) = x + \int_0^tX(\gamma(s))ds,
" * 
Main.indentation * raw"```
" * 
Main.indentation * raw"i.e. ``\Gamma`` maps ``C^\infty`` curves through ``x`` into ``C^\infty`` curves through ``x``. We further have with the norm ``||\gamma||_\infty = \mathrm{sup}_{t \in (-\epsilon, \epsilon)}|\gamma(t)|``:
" * 
Main.indentation * raw"```math
" *
Main.indentation * raw"\begin{aligned} 
" * 
Main.indentation * raw"||\Gamma(\gamma_1 - \gamma_2)||_\infty & = \mathrm{sup}_{t \in (-\epsilon, \epsilon)}\left| \int_0^t (X(\gamma_1(s)) - X(\gamma_2(s)))ds \right| \\
" * 
Main.indentation * raw"& \leq \mathrm{sup}_{t \in (-\epsilon, \epsilon)}\int_0^t | X(\gamma_1(s)) - X(\gamma_2(s)) | ds \\
" * 
Main.indentation * raw"& \leq \mathrm{sup}_{t \in (-\epsilon, \epsilon)}\int_0^t L |\gamma_1(s) - \gamma_2(s)| ds \\
" * 
Main.indentation * raw"& \leq \epsilon{}L \cdot \mathrm{sup}_{t \in (-\epsilon, \epsilon)}|\gamma_1(t) - \gamma_2(t)|,
" * 
Main.indentation * raw"\end{aligned}
" * 
Main.indentation * raw"```
" * 
Main.indentation * raw"and we see that ``\Gamma`` is a contractive mapping if we pick ``\epsilon`` small enough and we can hence apply the fixed-point theorem. So there has to exist a ``C^\infty`` curve through ``x`` that we call ``\gamma^*`` such that 
" * 
Main.indentation * raw"```math
" * 
Main.indentation * raw"\gamma^*(t) = \int_0^tX(\gamma^*(s))ds,
" *
Main.indentation * raw"and this ``\gamma^*`` is the curve we were looking for. Its uniqueness is guaranteed by the fixed-point theorem.")
``` 

For all the problems we discuss here we can extend the integral curves of ``X`` from the finite interval ``(-\epsilon, \epsilon)`` to all of ``\mathbb{R}``. The solution ``\gamma`` we call an *integral curve* or *flow* of the vector field (ODE).

## Time-Dependent Vector Fields

We proved the theorem above for a time-independent vector field ``X``, but it also holds for time-dependent vector fields, i.e. for mapping of the form: 

```math
X: [0,T]\times\mathcal{M}\to{}TM.
```

The proof for this case proceeds analogously to the case of the time-independent vector field; to apply the proof we simply have to *extend* the vector field to (here written for a specific coordinate chart ``\varphi_U``): 

```math
\bar{X}: [0, T]\times\mathbb{R}^n\to{}\mathbb{R}^{n+1},\, (t, x_1, \ldots, x_n) \mapsto (1, X(x_1, \ldots, x_n)).
```

More details on this can be found in e.g. [lang2012fundamentals](@cite). For `GeometricMachineLearning` time-dependent vector fields are important because many of the optimizers we are using (such as the [Adam optimizer](@ref "The Adam Optimizer")) can be seen as approximating the flow of a time-dependent vector field.

## Reference

```@bibliography
Pages = []
Canonical = false

lang2012real
lang2012fundamentals
```