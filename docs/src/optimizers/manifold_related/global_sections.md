# Global Sections

**Global sections** are needed needed for the generalization of [Adam](../adam_optimizer.md) and other optimizers to [homogeneous spaces](@ref "Homogeneous Spaces"). They are necessary to perform the two mappings represented represented by horizontal and vertical red lines in the section on the general [optimizer framework](../../Optimizer.md).

## Computing the global section
In differential geometry a **section** is always associated to some **bundle**, in our case this bundle is $\pi:G\to\mathcal{M},A\mapsto{}AE$. A section is a mapping $\mathcal{M}\to{}G$ for which $\pi$ is a left inverse, i.e. $\pi\circ\lambda = \mathrm{id}$. 

For the Stiefel manifold $St(n, N)\subset\mathbb{R}^{N\times{}n}$ we compute the global section the following way: 
1. Start with an element $Y\in{}St(n,N)$,
2. Draw a random matrix $A\in\mathbb{R}^{N\times{}(N-n)}$,
3. Remove the subspace spanned by $Y$ from the range of $A$: $A\gets{}A-YY^TA$
4. Compute a **QR decomposition** of $A$ and take as section $\lambda(Y) = [Y, Q_{[1:N, 1:(N-n)]}] =: [Y, \bar{\lambda}]$.

It is easy to check that $\lambda(Y)\in{}G=SO(N)$.

In `GeometricMachineLearning`, `GlobalSection` takes an element of $Y\in{}St(n,N)\equiv$`StiefelManifold{T}` and returns an instance of `GlobalSection{T, StiefelManifold{T}}`. The application $O(N)\times{}St(n,N)\to{}St(n,N)$ is done with the functions `apply_section!` and `apply_section`.

## Computing the global tangent space representation based on a global section

The output of the [horizontal lift](horizontal_lift.md) $\Omega$ is an element of $\mathfrak{g}^{\mathrm{hor},Y}$. For this mapping $\Omega(Y, B{}Y) = B$ if $B\in\mathfrak{g}^{\mathrm{hor},Y}$, i.e. there is **no information loss** and no projection is performed. We can map the $B\in\mathfrak{g}^{\mathrm{hor},Y}$ to $\mathfrak{g}^\mathrm{hor}$ with $B\mapsto{}\lambda(Y)^{-1}B\lambda(Y)$.

The function `global_rep` performs both mappings at once[^1], i.e. it takes an instance of `GlobalSection` and an element of $T_YSt(n,N)$, and then returns an element of $\frak{g}^\mathrm{hor}\equiv$`StiefelLieAlgHorMatrix`.

[^1]: For computational reasons.

In practice we use the following: 

```math
\begin{aligned}
\lambda(Y)^T\Omega(Y,\Delta)\lambda(Y)  & = \lambda(Y)^T[(\mathbb{I} - \frac{1}{2}YY^T)\Delta{}Y^T - Y\Delta^T(\mathbb{I} - \frac{1}{2}YY^T)]\lambda(Y) \\
                                        & = \lambda(Y)^T[(\mathbb{I} - \frac{1}{2}YY^T)\Delta{}E^T - Y\Delta^T(\lambda(Y) - \frac{1}{2}YE^T)] \\
                                        & = \lambda(Y)^T\Delta{}E^T - \frac{1}{2}EY^T\Delta{}E^T - E\Delta^T\lambda(Y) + \frac{1}{2}E\Delta^TYE^T \\ 
                                        & = \begin{bmatrix} Y^T\Delta{}E^T \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \frac{1}{2}EY^T\Delta{}E - \begin{bmatrix} E\Delta^TY & E\Delta^T\bar{\lambda} \end{bmatrix} + \frac{1}{2}E\Delta^TYE^T \\
                                        & = \begin{bmatrix} Y^T\Delta{}E^T \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} + E\Delta^TYE^T - \begin{bmatrix}E\Delta^TY & E\Delta^T\bar{\lambda} \end{bmatrix} \\
                                                & = EY^T\Delta{}E^T + E\Delta^TYE^T - E\Delta^TYE^T + \begin{bmatrix} \mathbb{O} \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O} & E\Delta^T\bar{\lambda} \end{bmatrix} \\
                                        & = EY^T\Delta{}E^T + \begin{bmatrix} \mathbb{O} \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O} & E\Delta^T\bar{\lambda} \end{bmatrix},
\end{aligned}
```

meaning that for an element of the [horizontal component of the Lie algebra](@ref "The Global Tangent Space for the Stiefel Manifold") ``\mathfrak{g}^\mathrm{hor}`` we store ``A=Y^T\Delta`` and ``B=\bar{\lambda}^T\Delta``.

## Optimization

The output of `global_rep` is then used for all the [optimization steps](../../Optimizer.md).

## References 

```@bibliography
Pages = []
Canonical = false

frankel2011geometry
```