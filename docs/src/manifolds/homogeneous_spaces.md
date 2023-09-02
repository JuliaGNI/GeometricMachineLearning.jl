# Homogeneous Spaces 

**Homogeneous spaces** are manifolds $\mathcal{M}$ on which a Lie group $G$ acts transitively, i.e.  

```math
\forall X,Y\in\mathcal{M} \exists{}A\in{}G\text{ s.t. }AX = Y.
```

Now fix a distinct element $E\in\mathcal{M}$. We can also establish an isomorphism between $\mathcal{M}$ and the quotient space $G/\sim$ with the equivalence relation: 
```math
A_1 \sim A_2 \iff A_1E = A_2E.
```
Note that this is independent of the chosen $E$.

The tangent spaces of $\mathcal{M}$ are of the form $T_Y\mathcal{M} = \mathfrak{g}\cdot{}Y$, i.e. can be fully described through its Lie algebra. 
Based on this we can perform a splitting of $\mathfrak{g}$ into two parts:

1. The **vertical component** $\mathfrak{g}^{\mathrm{ver},Y}$ is the kernel of the map $\mathfrak{g}\to{}T_Y\mathcal{M}, V \mapsto VY$, i.e. $\mathfrak{g}^{\mathrm{ver},Y} = \{V\in\mathfrak{g}:VY = 0\}.$

2. The **horizontal component** $\mathfrak{g}^{\mathrm{hor},Y}$ is the orthogonal complement of $\mathfrak{g}^{\mathrm{ver},Y}$ in $\mathfrak{g}$. It is isomorphic to $T_Y\mathcal{M}$.

We will refer to the mapping from $T_Y\mathcal{M}$ to $\mathfrak{g}^{\mathrm{hor}, Y}$ by $\Omega$. If we have now defined a metric $\langle\cdot,\cdot\rangle$ on $\mathfrak{g}$, then this induces a Riemannian metric on $\mathcal{M}$:
```math
g_Y(\Delta_1, \Delta_2) = \langle\Omega(Y,\Delta_1),\Omega(Y,\Delta_2)\rangle\text{ for $\Delta_1,\Delta_2\in{}T_Y\mathcal{M}$.}
```

Two examples of homogeneous spaces implemented in `GeometricMachineLearning` are the [Stiefel](stiefel_manifold.md) and the [Grassmann](grassmann_manifold.md) manifold.

## References 
- Frankel, Theodore. The geometry of physics: an introduction. Cambridge university press, 2011.