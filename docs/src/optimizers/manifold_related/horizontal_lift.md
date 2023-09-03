# The Horizontal Lift 

For each element $Y\in\mathcal{M}$ we can perform a splitting $\mathfrak{g} = \mathfrak{g}^{\mathrm{hor}, Y}\oplus\mathfrak{g}^{\mathrm{ver}, Y}$, where the two subspaces are the **horizontal** and the **vertical** component of $\mathfrak{g}$ at $Y$ respectively. For homogeneous spaces: $T_Y\mathcal{M} = \mathfrak{g}\cdot{}Y$, i.e. every tangent space to $\mathcal{M}$ can be expressed through the application of the Lie algebra to the relevant element. The vertical component consists of those elements of $\mathfrak{g}$ which are mapped to the zero element of $T_Y\mathcal{M}$, i.e. 

```math
\mathfrak{g}^{\mathrm{ver}, Y} := \mathrm{ker}(\mathfrak{g}\to{}T_Y\mathcal{M}).
```

The orthogonal complement[^1] of $\mathfrak{g}^{\mathrm{ver}, Y}$ is the horizontal component and is referred to by $\mathfrak{g}^{\mathrm{hor}, Y}$. This is naturally isomorphic to $T_Y\mathcal{M}$. For the Stiefel manifold the horizontal lift has the simple form: 

```math
\Omega(Y, V) = \left(\mathbb{I} - \frac{1}{2}\right)VY^T - YV^T(\mathbb{I} - \frac{1}{2}YY^T).
```

If the element $Y$ is the distinct element $E$, then the elements of $\mathfrak{g}^{\mathrm{hor},E}$ take a particularly simple form, see [Global Tangent Space](../../arrays/stiefel_lie_alg_horizontal.md) for a description of this. 


[^1]: The orthogonal complement is taken with respect to a metric defined on $\mathfrak{g}$. For the case of $G=SO(N)$ and $\mathfrak{g}=\mathfrak{so}(N) = \{A:A+A^T =0\}$ this metric can be chosen as $(A_1,A_2)\mapsto{}\frac{1}{2}A_1^TA_2$.
