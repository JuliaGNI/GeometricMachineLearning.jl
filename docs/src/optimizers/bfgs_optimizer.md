# The BFGS Optimizer

The presentation shown here is largely taken from [wright2006numerical; chapters 3 and 6](@cite) with a derivation based on an [online comment](https://math.stackexchange.com/questions/2091867/quasi-newton-methods-understanding-dfp-updating-formula). The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is a second order optimizer that can be also be used to train a neural network.

It is a version of a *quasi-Newton* method and is therefore especially suited for convex problems. As is the case with any other (quasi-)Newton method the BFGS algorithm approximates the objective with a quadratic function in each optimization step:
```math
m_k(x) = f(x_k) + (\nabla_{x_k}f)^T(x - x_k) + \frac{1}{2}(x - x_k)^TB_k(x - x_k),
```
where ``B_k`` is referred to as the *approximate Hessian*. We further require ``B_k`` to be symmetric and positive definite. Differentiating the above expression and setting the derivative to zero gives us: 
```math
\nabla_xm_k = \nabla_{x_k}f + B_k(x - x_k) = 0,
```
or written differently: 
```math
x - x_k = -B_k^{-1}\nabla_{x_k}f.
```
This value we will from now on call ``p_k := x - x_k`` and refer to as the *search direction*. The new iterate then is: 
```math
x_{k+1} = x_k + \alpha_kp_k,
```
where ``\alpha_k`` is the *step length*. Techniques that describe how to pick an appropriate ``\alpha_k`` are called *line-search methods* and are discussed below. First we discuss what requirements we impose on ``B_k``. A first reasonable condition would be to require the gradient of ``m_k`` to be equal to that of ``f`` at the points ``x_{k-1}`` and ``x_k``: 
```math
\begin{aligned}
\nabla_{x_k}m_k  & = \nabla_{x_k}f + B_k(x_k - x_k)  & \overset{!}{=} \nabla_{x_k}f \text{ and } \\
\nabla_{x_{k-1}}m_k & = \nabla{x_k}f + B_k(x_{k-1} - x_k) & \overset{!}{=} \nabla_{x_{k-1}}f.
\end{aligned}
```
The first one of these conditions is of course automatically satisfied. The second one can be rewritten as: 
```math
B_k(x_k - x_{k-1}) = \overset{!}{=} \nabla_{x_k}f - \nabla_{x_{k-1}}f. 
```

The following notations are often used: 
```math
s_{k-1} := \alpha_{k-1}p_{k-1} :=  x_{k} - x_{k-1} \text{ and } y_{k-1} := \nabla_{x_k}f - \nabla_{x_{k-1}}f. 
```

The conditions mentioned above then becomes: 
```math
B_ks_{k-1} \overset{!}{=} y_{k-1},
```
and we call it the *secant equation*. A second condition we impose on ``B_k`` is that is has to be positive-definite at point ``s_{k-1}``:
```math
s_{k-1}^Ty_{k-1} > 0.
```
This is referred to as the *curvature condition*. If we impose the *Wolfe conditions*, the *curvature condition* hold automatically. The Wolfe conditions are stated with respect to the parameter ``\alpha_k``.

The *Wolfe conditions* are:
1. ``f(x_k+\alpha{}p_k)\leq{}f(x_k) + c_1\alpha(\nabla_{x_k}f)^Tp_k`` for ``c_1\in(0,1)``.
2. ``(\nabla_{(x_k + \alpha_kp_k)}f)^Tp_k \geq c_2(\nabla_{x_k}f)^Tp_k`` for ``c_2\in(c_1,1)``.

A possible choice for ``c_1`` and ``c_2`` are ``10^{-4}`` and ``0.9`` (see [wright2006numerical](@cite)). The two Wolfe conditions above are respectively called the *sufficient decrease condition* and the *curvature condition* respectively. Note that the second Wolfe condition (also called curvature condition) is stronger than the one mentioned before under the assumption that the first Wolfe condition is true:
```math
(\nabla_{x_k}f)^Tp_{k-1} - c_2(\nabla_{x_{k-1}}f)^Tp_{k-1} = y_{k-1}^Tp_{k-1} + (1 - c_2)(\nabla_{x_{k-1}}f)^Tp_{k-1} \geq 0,
```
and the second term in this expression is ``(1 - c_2)(\nabla_{x_{k-1}}f)^Tp_{k-1}\geq\frac{1-c_2}{c_1\alpha_{k-1}}(f(x_k) - f(x_{k-1}))``, which is negative. 

In order to pick the ideal ``B_k`` we solve the following problem: 
```math
\begin{aligned}
\min_B & ||B - B_{k-1}||_W \\ 
\text{s.t.} & B  = B^T\text{ and }Bs_{k-1}=y_{k-1},
\end{aligned}
```
where the first condition is symmetry and the second one is the secant equation. For the norm ``||\cdot||_W`` we pick the weighted Frobenius norm:
```math
||A||_W := ||W^{1/2}AW^{1/2}||_F,
```
where ``||\cdot||_F`` is the usual Frobenius norm[^1] and the matrix ``W=\tilde{B}_{k-1}`` is the inverse of the *average Hessian*:
```math
\tilde{B}_{k-1} = \int_0^1 \nabla^2f(x_{k-1} + \tau\alpha_{k-1}p_{k-1})d\tau.
``` 
[^1]: The Frobenius norm is ``||A||_F^2 = \sum_{i,j}a_{ij}^2``.

In order to find the ideal ``B_k`` under the conditions described above, we introduce some notation: 
- ``\tilde{B}_{k-1} := W^{1/2}B_{k-1}W^{1/2}``,
- ``\tilde{B} := W^{1/2}BW^{1/2}``, 
- ``\tilde{y}_{k-1} := W^{1/2}y_{k-1}``, 
- ``\tilde{s}_{k-1} := W^{-1/2}s_{k-1}``.

With this notation we can rewrite the problem of finding ``B_k`` as: 
```math
\begin{aligned}
\min_{\tilde{B}} & ||\tilde{B} - \tilde{B}_{k-1}||_F \\ 
\text{s.t.} & \tilde{B} = \tilde{B}^T\text{ and }\tilde{B}\tilde{s}_{k-1}=\tilde{y}_{k-1}.
\end{aligned}
```

We further have ``Wy_{k-1} = s_{k-1}`` (by the mean value theorem ?) and therefore ``\tilde{y}_{k-1} = \tilde{s}_{k-1}``.

Now we rewrite ``B`` and ``B_{k-1}`` in a new basis ``U = [u|u_\perp]``, where ``u := \tilde{s}_{k-1}/||\tilde{s}_{k-1}||`` and ``u_perp`` is an orthogonal complement[^2] of ``u``:

[^2]: So we must have ``u^Tu_\perp=0`` and further ``u_\perp^Tu_\perp=\mathbb{I}``.

```math
\begin{aligned}
U^T\tilde{B}_{k-1}U - U^T\tilde{B}U = \begin{bmatrix}  u^T \\ u_\perp^T \end{bmatrix}(\tilde{B}_{k-1} - \tilde{B})\begin{bmatrix} u & u_\perp \end{bmatrix} = \\
\begin{bmatrix}
    u^T\tilde{B}_{k-1}u - 1 & u^T\tilde{B}_{k-1}u \\
    u_\perp^T\tilde{B}_{k-1}u & u_\perp^T(\tilde{B}_{k-1}-\tilde{B}_k)u_\perp
\end{bmatrix}.
\end{aligned}
```
By a property of the Frobenius norm: 
```math
||\tilde{B}_{k-1} - \tilde{B}||^2_F = (u^T\tilde{B}_{k-1} -1)^2 + ||u^T\tilde{B}_{k-1}u_\perp||_F^2 + ||u_\perp^T\tilde{B}_{k-1}u||_F^2 + ||u_\perp^T(\tilde{B}_{k-1} - \tilde{B})u_\perp||_F^2.
```

We see that ``\tilde{B}`` only appears in the last term, which should therefore be made zero. This then gives: 
```math
\tilde{B} = U\begin{bmatrix} 1 & 0 \\ 0 & u^T_\perp\tilde{B}_{k-1}u_\perp \end{bmatrix} = uu^T + (\mathbb{I}-uu^T)\tilde{B}_{k-1}(\mathbb{I}-uu^T).
```

If we now map back to the original coordinate system, the ideal solution for ``B_k`` is: 
```math
B_k = (\mathbb{I} - \frac{1}{y_{k-1}^Ts_{k-1}}y_{k-1}s_{k-1}^T)B_{k-1}(\mathbb{I} - \frac{1}{y_{k-1}^Ts_{k-1}}s_{k-1}y_{k-1}^T) + \frac{1}{y_{k-1}^Ts_{k-1}}y_ky_k^T.
```

What we need in practice however is not ``B_k``, but its inverse ``H_k``. This is because we need to find ``s_{k-1}`` based on ``y_{k-1}``.  To get ``H_k`` based on the expression for ``B_k`` above we can use the *Sherman-Morrison-Woodbury formula*[^3] to obtain:

[^3]: The *Sherman-Morrison-Woodbury formula* states ``(A + UCV)^{-1} = A^{-1} - A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}``.

```math
H_{k} = H_{k-1} - \frac{H_{k-1}y_{k-1}y_{k-1}^TH_{k-1}}{y_{k-1}^TH_{k-1}y_{k-1}} + \frac{s_{k-1}s_{k-1}^T}{y_{k-1}^Ts_{k-1}}.
```


TODO: Example where this works well!

## References 

```@bibliography
Pages = []
Canonical = false 

wright2006numerical
```