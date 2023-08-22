# Retractions

Retractions are a map from the ``horizontal part'' of the Lie algebra $\mathfrak{g}^\mathrm{hor}$ to the respective manifold (homogeneous space).

Homogeneous spaces (i.e. the all the manifolds treated in `GeometricMachineLearning`) have the structure $\mathcal{M} = G/\sim$, i.e. are a Lie group modulu an equivalence relation.  For us this equivalence relation is: two elements $A_1$ and $A_2$ are equivalent ($A_1 \sim A_2$) iff their application to the *canonical element* $E\in\mathcal{M}$ is the same, i.e. $A_1E = A_2E$. 

For the Stiefel manifold $St(n,N)$ this canonical element is 
```math
E = \begin{bmatrix}
    \mathbb{I}  \\
    \mathbb{O}
\end{bmatrix},
```
where the matrices in the first row are $\in\mathbb{R}^{n\times{}n}$ and the matrices in the second row are $\in\mathbb{R}^{(N-n)\times{}n}$.

For optimization in neural networks (almost always first order) we solve a gradient flow equation $\dot{W} = -\eta\cdot\mathrm{grad}_WL$, where $\mathrm{grad}_WL$ is the *Riemannian gradient* of the loss function $L$ evaluated at position $W$.

If we deal with Euclidean spaces (vector spaces), then this gradient is just the result of an AD routine and we do not have to do anything else. 

For manifolds, after we obtained the Riemannian gradient (see e.g. the section on Stiefel manifold for how this is done there), we have to solve a *geodesic equation*. This is a canonical ODE associated with any Riemannian manifold. 

The general theory of Riemannian manifolds is rather complicated, but for the neural networks treated in `GeometricMachineLearning`, we only rely on optimization of matrix Lie groups and homogeneous spaces, which is much simpler. 

For Lie groups each tangent space is isomorphic to its Lie algebra $\mathfrak{g}\equiv{}T_\mathbb{I}G$. The geodesic map from $\mathfrak{g}$ to $G$, for matrix Lie groups with bi-invariant Riemannian metric, is simply the application of the matrix exponential $\exp$. Alternatively this can be replaced by the Cayley transform (see (Absil et al, 2008).)
 
Starting from this basic map $\exp:\mathfrak{g}\to{}G$ we can build mappings for more complicated cases: 

1. General tangent space to a Lie group $T_AG$: The geodesic map for an element $V\in{}T_AG$ is simply $A\exp(A^{-1}V)$.

2. Special tangent space to a homogeneous space $T_E\mathcal{M}$: For $V=BE\in{}T_E\mathcal{M}$ the exponential map is simply $\exp(B)E$. 

3. General tangent space to a homogeneous space $T_Y\mathcal{M}$ for $Y = AE$: For $V=ABE\in{}T_Y\mathcal{M}$ with $Y = AE$ the exponential map is simply $A\exp(B)E$. This is the general case with which we deal with.  

What retraction in `GeometricMachineLearning` does is the map $\mathfrak{g}^\mathrm{hor}\to\mathcal{M}$, which is the second of the above points. To get the third from the second point, we simply have to multiply with a matrix. 


### Word of caution

Note that the $O(N)$, the Lie group corresponding to the Stiefel manifold, has a bi-invariant Riemannian metric associated with it: $(B_1,B_2)\mapsto \mathrm{Tr}(B_1^TB_2)$.
For other Lie groups (e.g. the symplectic group) the situation is slightly more difficult (see (Bendokat et al, 2021).)

## References 

- Absil P A, Mahony R, Sepulchre R. Optimization algorithms on matrix manifolds[M]. Princeton University Press, 2008.

- Bendokat T, Zimmermann R. The real symplectic Stiefel and Grassmann manifolds: metrics, geodesics and applications[J]. arXiv preprint arXiv:2108.12447, 2021.

- Brantner B. Generalizing Adam To Manifolds For Efficiently Training Transformers[J]. arXiv preprint arXiv:2305.16901, 2023.