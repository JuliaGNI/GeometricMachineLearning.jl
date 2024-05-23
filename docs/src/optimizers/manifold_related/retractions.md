# Retractions

## Classical Definition
Classically, retractions are defined as maps smooth maps 

```math
R: T\mathcal{M}\to\mathcal{M}:(x,v)\mapsto{}R_x(v)
```

such that each curve $c(t) := R_x(tv)$ satisfies $c(0) = x$ and $c'(0) = v$.

## In `GeometricMachineLearning`

Retractions are a map from the **horizontal component** of the Lie algebra $\mathfrak{g}^\mathrm{hor}$ to the respective manifold.

For optimization in neural networks (almost always first order) we solve a gradient flow equation 

```math
\dot{W} = -\mathrm{grad}_WL, 
```
where $\mathrm{grad}_WL$ is the **Riemannian gradient** of the loss function $L$ evaluated at position $W$.

If we deal with Euclidean spaces (vector spaces), then the Riemannian gradient is just the result of an AD routine and the solution of the equation above can be approximated with $W^{t+1} \gets W^t - \eta\nabla_{W^t}L$, where $\eta$ is the **learning rate**. 

For manifolds, after we obtained the Riemannian gradient (see e.g. the section on [Stiefel manifold](@ref "The Stiefel Manifold")), we have to solve a **geodesic equation**. This is a canonical ODE associated with any Riemannian manifold. 

The general theory of Riemannian manifolds is rather complicated, but for the neural networks treated in `GeometricMachineLearning`, we only rely on optimization of matrix Lie groups and [homogeneous spaces](../../manifolds/homogeneous_spaces.md), which is much simpler. 

For Lie groups each tangent space is isomorphic to its Lie algebra $\mathfrak{g}\equiv{}T_\mathbb{I}G$. The geodesic map from $\mathfrak{g}$ to $G$, for matrix Lie groups with bi-invariant Riemannian metric like $SO(N)$, is simply the application of the matrix exponential $\exp$. Alternatively this can be replaced by the Cayley transform (see (Absil et al, 2008).)
 
Starting from this basic map $\exp:\mathfrak{g}\to{}G$ we can build mappings for more complicated cases: 

1. **General tangent space to a Lie group** $T_AG$: The geodesic map for an element $V\in{}T_AG$ is simply $A\exp(A^{-1}V)$.

2. **Special tangent space to a homogeneous space** $T_E\mathcal{M}$: For $V=BE\in{}T_E\mathcal{M}$ the exponential map is simply $\exp(B)E$. 

3. **General tangent space to a homogeneous space** $T_Y\mathcal{M}$ with $Y = AE$: For $\Delta=ABE\in{}T_Y\mathcal{M}$ the exponential map is simply $A\exp(B)E$. This is the general case which we deal with.  

The general theory behind points 2. and 3. is discussed in chapter 11 of (O'Neill, 1983). The function `retraction` in `GeometricMachineLearning` performs $\mathfrak{g}^\mathrm{hor}\to\mathcal{M}$, which is the second of the above points. To get the third from the second point, we simply have to multiply with a matrix from the left. This step is done with `apply_section` and represented through the red vertical line in the diagram on the [general optimizer framework](../../Optimizer.md).


### Word of caution

The Lie group corresponding to the Stiefel manifold $SO(N)$ has a bi-invariant Riemannian metric associated with it: $(B_1,B_2)\mapsto \mathrm{Tr}(B_1^TB_2)$.
For other Lie groups (e.g. the symplectic group) the situation is slightly more difficult (see (Bendokat et al, 2021).)

## References 

- Absil P A, Mahony R, Sepulchre R. Optimization algorithms on matrix manifolds[M]. Princeton University Press, 2008.

- Bendokat T, Zimmermann R. The real symplectic Stiefel and Grassmann manifolds: metrics, geodesics and applications[J]. arXiv preprint arXiv:2108.12447, 2021.

- O'Neill, Barrett. Semi-Riemannian geometry with applications to relativity. Academic press, 1983.