# The BFGS Optimizer

The presentation shown here is largely taken from [wright2006numerical; chapters 3 and 6](@cite) with a derivation based on an online comment [2279304](@cite). The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is a second order optimizer that can be also be used to train a neural network.

It is a version of a *quasi-Newton* method and is therefore especially suited for convex problems. As is the case with any other (quasi-)Newton method[^1] the BFGS algorithm approximates the objective with a quadratic function in each optimization step:

[^1]: Various Newton methods and quasi-Newton methods differ in how they model the *approximate Hessian*.

```math
m^{(k)}(x) = L(x^{(k)}) + (\nabla_{x^{(k)}}L)^T(x - x^{(k)}) + \frac{1}{2}(x - x^{(k)})^TR^{(k)}(x - x^{(k)}),
```
where ``R^{(k)}`` is referred to as the *approximate Hessian*. We further require ``R^{(k)}`` to be symmetric and positive definite. Differentiating the above expression and setting the derivative to zero gives us: 
```math
\nabla_xm^{(k)} = \nabla_{x^{(k)}}L + R^{(k)}(x - x^{(k)}) = 0,
```
or written differently: 
```math
x - x^{(k)} = -(R^{(k)})^{-1}\nabla_{x^{(k)}}L.
```
This value we will from now on call ``p^{(k)} := -H^{(k)}\nabla_{x_k}L`` with ``H^{(k)} := (R^{(k)})^{-1}`` and refer to as the *search direction*. The new iterate then is: 
```math
x^{(k+1)} = x^{(k)} + \eta^{(k)}p^{(k)},
```
where ``\eta^{(k)}`` is the *step length*. Techniques that describe how to pick an appropriate ``\eta^{(k)}`` are called *line-search methods* and are discussed below. First we discuss what requirements we impose on ``R^{(k)}``. A first reasonable condition would be to require the gradient of ``m^{(k)}`` to be equal to that of ``L`` at the points ``x^{(k-1)}`` and ``x^{(k)}``: 
```math
\begin{aligned}
\nabla_{x^{(k)}}m^{(k)}  & = \nabla_{x^{(k)}}L + R^{(k)}(x^{(k)} - x^{(k)})  & \overset{!}{=} & \nabla_{x^{(k)}}L  & \text{ and } \\
\nabla_{x^{(k-1)}}m^{(k)} & = \nabla_{x^{(k)}}L + R^{(k)}(x^{(k-1)} - x^{(k)}) & \overset{!}{=} & \nabla_{x^{(k-1)}}L. & 
\end{aligned}
```
The first one of these conditions is automatically satisfied. The second one can be rewritten as: 
```math
x^{(k)} - x^{(k-1)} \overset{!}{=} H^{(k)}(\nabla_{x^{(k)}}L - \nabla_{x^{(k-1)}}L). 
```

The following notations are often used: 
```math
s^{(k-1)} := \eta^{(k-1)}p^{(k-1)} :=  x^{(k)} - x^{(k-1)} \quad\text{ and }\quad y^{(k-1)} := \nabla_{x^(k)}L - \nabla_{x^{(k-1)}}L.
```

The condition mentioned above then becomes: 
```math
s^{(k-1)} \overset{!}{=} H^{(k)}y^{(k-1)},
```
and we call it the *secant equation*. 

In order to pick the ideal ``H^{(k)}`` we solve the following problem: 
```math
\begin{aligned}
& \min_H & & ||H - H^{(k-1)}||_W \\ 
& \text{s.t.} & & H = H^T\quad\text{and}\\
            & \text{and} & & s^{(k-1)} = Hy^{(k-1)},
\end{aligned}
```
where the first condition is symmetry and the second one is the secant equation. For the norm ``||\cdot||_W`` we pick the weighted Frobenius norm:
```math
||A||_W := ||W^{1/2}AW^{1/2}||_F,
```
where ``||\cdot||_F`` is the usual Frobenius norm[^2] and the matrix ``W=\tilde{R}^{(k-1)}`` is the *average Hessian*:
```math
\tilde{R}^{(k-1)} = \int_0^1 \nabla^2f(x^{(k-1)} + \tau\eta^{(k-1)}p^{(k-1)})d\tau.
``` 
[^2]: The Frobenius norm is ``||A||_F^2 = \sum_{i,j}a_{ij}^2``.

We now state the solution to this minimization problem:

```@eval
Main.theorem(raw"The solution of the minimization problem is:
" * Main.indentation * raw"```math
" * Main.indentation * raw"H^{(k)} = (\mathbb{I} - \frac{1}{(s^{(k-1)})^Ty^{(k-1)}}s^{(k-1)}(y^{(k-1)})^T)H^{(k-1)}(\mathbb{I} - \frac{1}{(s^{k-1})^Ty^{(k-1)}}y^{(k-1)}(s^{(k-1)})^T) + \\ \frac{1}{(s^{(k-1)})^Ty^{(k-1)}}s^{(k-1)}(s^{(k-1)})^T,
" * Main.indentation * raw"```
" * Main.indentation * raw"with ``y^{(k-1)} = \nabla_{x^{(k)}}L - \nabla_{x^{(k-1)}}L`` and ``s^{(k-1)} = x^{(k)} - x^{(k-1)}`` as above.")
```

```@eval
Main.proof(raw"In order to find the ideal ``H^{(k)}`` under the conditions described above, we introduce some notation: 
" * Main.indentation * raw"- ``\tilde{H}^{(k-1)} := W^{1/2}H^{(k-1)}W^{1/2}``,
" * Main.indentation * raw"- ``\tilde{H} := W^{1/2}HW^{1/2}``, 
" * Main.indentation * raw"- ``\tilde{y}^{(k-1)} := W^{-1/2}y^{(k-1)}``, 
" * Main.indentation * raw"- ``\tilde{s}^{(k-1)} := W^{1/2}s^{(k-1)}``.
" * Main.indentation * raw"
" * Main.indentation * raw"With this notation we can rewrite the problem of finding ``H^{(k)}`` as: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\begin{aligned}
" * Main.indentation * raw"& \min_{\tilde{H}} & & ||\tilde{H} - \tilde{H}^{(k-1)}||_F  \\ 
" * Main.indentation * raw"& \text{s.t.}\quad & & \tilde{H} = \tilde{H}^T\quad \\
" * Main.indentation * raw"& \text{and} & & \tilde{s}^{(k-1)} = \tilde{H}\tilde{y}^{(k-1)}.
" * Main.indentation * raw"\end{aligned}
" * Main.indentation * raw"```
" * Main.indentation * raw"
" * Main.indentation * raw"We further have ``y^{(k-1)} = Ws^{(k-1)}`` and hence ``\tilde{y}^{(k-1)} = \tilde{s}^{(k-1)}`` by a corollary of the mean value theorem: ``\int_0^1 g'(\xi_1 + \tau(\xi_2 - \xi_1)) d\tau (\xi_2 - \xi_1) = g(\xi_2) - g(\xi_1)`` for a vector-valued function ``g``.
" * Main.indentation * raw"
" * Main.indentation * raw"Now we rewrite ``H`` and ``H^{(k-1)}`` in a new basis ``U = [u|u_\perp]``, where ``u := \tilde{y}^{(k-1)}/||\tilde{y}^{(k-1)}||`` and ``u_\perp`` is an orthogonal complement of ``u`` (i.e. we have ``u^Tu_\perp=0`` and ``u_\perp^Tu_\perp=\mathbb{I}``):
" * Main.indentation * raw"
" * Main.indentation * raw"```math
" * Main.indentation * raw"\begin{aligned}
" * Main.indentation * raw"U^T\tilde{H}^{(k-1)}U - U^T\tilde{H}U = \begin{bmatrix}  u^T \\ u_\perp^T \end{bmatrix}(\tilde{H}^{(k-1)} - \tilde{H})\begin{bmatrix} u & u_\perp \end{bmatrix} = \\
" * Main.indentation * raw"\begin{bmatrix}
" * Main.indentation * raw"    u^T\tilde{H}^{(k-1)}u - 1 & u^T\tilde{H}^{(k-1)}u_\perp \\
" * Main.indentation * raw"    u_\perp^T\tilde{H}^{(k-1)}u & u_\perp^T(\tilde{H}^{(k-1)}-\tilde{H}^{(k)})u_\perp
" * Main.indentation * raw"\end{bmatrix}.
" * Main.indentation * raw"\end{aligned}
" * Main.indentation * raw"```
" * Main.indentation * raw"By a property of the Frobenius norm we can consider the blocks independently: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\begin{aligned}
" * Main.indentation * raw"||\tilde{H}^{(k-1)} - \tilde{H}||^2_F & = ||U^T(\tilde{H}^{(k-1)} - \tilde{H})U||^2_F \\
" * Main.indentation * raw"& = (u^T\tilde{H}^{(k-1)}u -1)^2 + ||u^T\tilde{H}^{(k-1)}u_\perp||_F^2 + ||u_\perp^T\tilde{H}^{(k-1)}u||_F^2 + ||u_\perp^T(\tilde{H}^{(k-1)} - \tilde{H})u_\perp||_F^2.
" * Main.indentation * raw"\end{aligned}
" * Main.indentation * raw"```
" * Main.indentation * raw"We see that ``\tilde{H}`` only appears in the last term, which should therefore be made zero, i.e. the projections of ``\tilde{H}_{k-1}`` and ``\tilde{H}`` onto the space spanned by ``u_\perp`` should coincide. With the condition ``\tilde{H}u \overset{!}{=} u`` we hence get: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\tilde{H} = U\begin{bmatrix} 1 & 0 \\ 0 & u^T_\perp\tilde{H}^{(k-1)}u_\perp \end{bmatrix}U^T = uu^T + (\mathbb{I}-uu^T)\tilde{H}^{(k-1)}(\mathbb{I}-uu^T).
" * Main.indentation * raw"```
" * Main.indentation * raw"If we now map back to the original coordinate system, the ideal solution for ``H^{(k)}`` is: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"H^{(k)} = (\mathbb{I} - \frac{1}{(s^{(k-1)})^Ty^{(k-1)}}s^{(k-1)}(y^{(k-1)})^T)H^{(k-1)}(\mathbb{I} - \frac{1}{(s^{k-1})^Ty^{(k-1)}}y^{(k-1)}(s^{(k-1)})^T) + \\ \frac{1}{(s^{(k-1)})^Ty^{(k-1)}}s^{(k-1)}(s^{(k-1)})^T,
" * Main.indentation * raw"```
" * Main.indentation * raw"and the assertion is proved.")
```

The cache and the parameters are updated with:
1. Compute the gradient ``\nabla_{x^{(k)}}L``,
2. obtain a negative search direction ``p^{(k)} \gets -H^{(k)}\nabla_{x^{(k)}}L``,
3. compute ``s^{(k)} = \eta^{(k)}p^{(k)}``,
4. compute ``y^{(k)} \gets \nabla_{x^{(k)}}L - \nabla_{x^{(k-1)}}L``,
5. update ``H^{(k + 1)} \gets (\mathbb{I} - \frac{1}{(s^{(k)})^Ty^{(k)}}s^{(k)}(y^{(k)})^T)H^{(k)}(\mathbb{I} - \frac{1}{s^({k})^Ty^{(k)}}y^{(k)}(s^{(k)})^T) + \\ \frac{1}{(s^{(k)})^Ty^{(k)}}s^{(k)}(s^{(k)})^T``,
6. update ``x^{(k + 1)} \gets x^{(k)} + s^{(k)}``.

The cache of the BFGS algorithm thus consists of the matrix ``H^{(\cdot)}`` for each weight ``x^{(\cdot)}`` in the neural network and the gradient for the previous time step ``\nabla_{x^{(k-1)}}L``. ``s^{(k)}`` here is again the *velocity* that we use to update the neural network weights. 

## The Riemannian Version of the BFGS Algorithm

Generalizing the BFGS algorithm to the setting of a Riemannian manifold is straightforward. All we have to do is replace Euclidean gradient by Riemannian ones (composed with a lift via [`global_rep`](@ref)): 

```math
\nabla_{x^{(k)}}L \implies (\Lambda^{(k)})^{-1}(\Omega(x^{(k)}, \mathrm{grad}_{x^{(k)}}L))\Lambda^{(k)} = \mathtt{global\_rep}(\mathrm{grad}_{x^{(k)}}),
```

and addition by a retraction:

```math
    x^{(k+1)} \gets x^{(k)} + s^{(k)} \implies x^{(k+1)} \gets \mathrm{Retraction}(s^{(k)})x^{(k)}.
```

The Hessian for the manifold BFGS algorithm is of size ``\tilde{N}\times\tilde{N}`` where ``\tilde{N} = \mathrm{dim}(\mathfrak{g}^\mathrm{hor})``. For [the global tangent space belonging to the Stiefel manifold](@ref "The Global Tangent Space for the Stiefel Manifold") we have ``\tilde{N} = (N - n)n + n(n - 1)\div2``.

In order to multiply the stored weights with the Hessian ``H`` we use the vectorization operation [`vec`](@ref) for all weights:

```@example bfgs_cache
using GeometricMachineLearning # hide
A = SkewSymMatrix([1, 2, 3], 3)
B = [4 5 6; ]
B̄ = StiefelLieAlgHorMatrix(A, B, 4, 3)
B̄ |> vec
```

The ``H`` matrix in the cache is correspondingly initialized as:

```@example bfgs_cache
@assert BFGSCache(B̄).H == [1  0  0  0  0  0; 0  1  0  0  0  0; 0  0  1  0  0  0; 0  0  0  1  0  0; 0  0  0  0  1  0; 0  0  0  0  0  1] # hide
BFGSCache(B̄)
```

We see that ``\bar{B}`` is of dimension ``\tilde{N} = (N - n)n + n(n - 1)\div2 = 3 + 3 = 6`` and ``H`` is of dimension ``\tilde{N}\times\tilde{N} = 6\times6.``

## The Curvature Condition and the Wolfe Conditions

In textbooks [wright2006numerical](@cite) an application of the BFGS algorithm typically further involves a line search subject to the *Wolfe conditions*. If these are satisfied the *curvature condition* usually also is.

A condition that is similar to the *secant condition* discussed before is that ``R^{(k)}`` has to be positive-definite at point ``s^{(k-1)}``:
```math
(s^{(k-1)})^Ty^{(k-1)} > 0.
```
This is referred to as the *standard curvature condition*. If we impose the *Wolfe conditions*, the *standard curvature condition* holds automatically. The Wolfe conditions are stated with respect to the parameter ``\eta^{(k)}``.

```@eval
Main.definition(raw"The **Wolfe conditions** are:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\begin{aligned}
" * Main.indentation * raw"L(x^{(k)}+\eta^{(k)}p^{(k)}) & \leq{}L(x^{(k)}) + c_1\eta^{(k)}(\nabla_{x^{(k)}}L)^Tp^{(k)} & \text{ for } & c_1\in(0,1) \quad\text{and} \\
" * Main.indentation * raw"(\nabla_{(x^{(k)} + \eta^{(k)}p^{(k)})}L)^Tp^{(k)} & \geq c_2(\nabla_{x^{(k)}}L)^Tp^{(k)} & \text{ for } & c_2\in(c_1,1).
" * Main.indentation * raw"\end{aligned}
" * Main.indentation * raw"```
" * Main.indentation * raw"The two Wolfe conditions above are respectively called the *sufficient decrease condition* and the *curvature condition*.")
```

A possible choice for ``c_1`` and ``c_2`` are ``10^{-4}`` and ``0.9`` [wright2006numerical](@cite). We further have:

```@eval
Main.theorem(raw"The second Wolfe condition, also called curvature condition, is stronger than the standard curvature condition under the assumption that the first Wolfe condition is true and ``L(x^{(k+1)}) < L(^{(x_k)})``.")
```

```@eval
Main.proof(raw"We use the second Wolfe condition to write
" * Main.indentation * raw"```math
" * Main.indentation * raw"(\nabla_{x^{(k)}}L)^Tp^{(k-1)} - c_2(\nabla_{x^{(k-1)}}L)^Tp^{(k-1)} = (y^{(k-1)})^Tp^{(k-1)} + (1 - c_2)(\nabla_{x^{(k-1)}}L)^Tp^{(k-1)} \geq 0,
" * Main.indentation * raw"```
" * Main.indentation * raw"and we can apply the first Wolfe condition on the second term in this expression: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"(1 - c_2)(\nabla_{x^{(k-1)}}L)^Tp^{(k-1)}\geq\frac{1-c_2}{c_1\eta^{(k-1)}}(L(x^{(k)}) - L(x^{(k-1)})),
" * Main.indentation * raw"```
" * Main.indentation * raw"which is negative if the value of ``L`` is decreasing.")
```

It is noteworthy that line search has not been used a lot in deep learning in the past. This is beginning to change however [kenneweg2024improving, vaswani2019painless](@cite).

## Stability of the Algorithm

Similar to the [Adam optimizer](@ref "The Adam Optimizer") we also add a ``\delta`` term for stability to two of the terms appearing in the update rule of the BFGS algorithm in practice. 

## Library Functions

```@docs
BFGSOptimizer
BFGSCache
update!(::Optimizer{<:BFGSOptimizer}, ::BFGSCache, ::AbstractArray)
vec(::StiefelLieAlgHorMatrix)
```

```@raw latex
section*{Chapter Summary}

In this chapter we gave explicit examples of neural network optimizers and demonstrated the corresponding application interface; we referred to the corresponding updating rules as \textit{optimizer methods}. A central part for all optimizers was the \textit{optimizer cache}. This depends on the specific optimizer method and has to be parallel transported along the optimization trajectory; for vector spaces this is trivial and for homogeneous spaces this is done by utilizing the \textit{global tangent space representation}. 
```

## References 

```@bibliography
Pages = []
Canonical = false 

wright2006numerical
2279304
huang2016riemannian
```