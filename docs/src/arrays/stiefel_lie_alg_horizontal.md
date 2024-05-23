# Horizontal component of the Lie algebra $\mathfrak{g}$

What we use to optimize Adam (and other algorithms) to manifolds is a **global tangent space representation** of the homogeneous spaces. 

For the [Stiefel manifold](@ref "The Stiefel Manifold"), this global tangent space representation takes a simple form: 

```math 
\mathcal{B} = \begin{bmatrix}
    A & -B^T \\ 
    B & \mathbb{O}
\end{bmatrix},
```

where ``A\in\mathbb{R}^{n\times{}n}`` is skew-symmetric and ``B\in\mathbb{R}^{N\times{}n}`` is arbitary. In `GeometricMachineLearning` the struct `StiefelLieAlgHorMatrix` implements elements of this form.

## Theoretical background

### Vertical and horizontal components

The Stiefel manifold ``St(n, N)`` is a [homogeneous space](@ref "Homogeneous Spaces") obtained from ``SO(N)`` by setting two matrices, whose first $n$ columns conincide, equivalent. 
Another way of expressing this is: 
```math
A_1 \sim A_2 \iff A_1E = A_2E
```
for 
```math 
E = \begin{bmatrix} \mathbb{I} \\ \mathbb{O}\end{bmatrix}.
```

Because ``St(n,N)`` is a homogeneous space, we can take any element ``Y\in{}St(n,N)`` and ``SO(N)`` acts transitively on it, i.e. can produce any other element in ``SO(N)``. A similar statement is also true regarding the tangent spaces of ``St(n,N)``, namely: 

```math
T_YSt(n,N) = \mathfrak{g}\cdot{}Y,
```

i.e. every tangent space can be expressed through an action of the associated Lie algebra. 

The kernel of the mapping $\mathfrak{g}\to{}T_YSt(n,N), B\mapsto{}BY$ is referred to as $\mathfrak{g}^{\mathrm{ver},Y}$, the **vertical component** of the Lie algebra at $Y$. In the case ``Y=E`` it is easy to see that elements belonging to $\mathfrak{g}^{\mathrm{ver},E}$ are of the following form: 
```math 
\begin{bmatrix}
\hat{\mathbb{O}} & \tilde{\mathbb{O}}^T \\ 
\tilde{\mathbb{O}} & C
\end{bmatrix},
```
where $\hat{\mathbb{O}}\in\mathbb{R}^{n\times{}n}$ is a "small" matrix and $\tilde{\mathbb{O}}\in\mathbb{R}^{N\times{}n}$ is a bigger one. $C\in\mathbb{R}^{N\times{}N}$ is a skew-symmetric matrix. 

The *orthogonal complement* of the vertical component is referred to as the **horizontal component** and denoted by ``\mathfrak{g}^{\mathrm{hor}, Y}``. It is isomorphic to ``T_YSt(n,N)`` and this isomorphism can be found explicitly. In the case of the Stiefel manifold: 

```math
\Omega(Y, \cdot):T_YSt(n,N)\to\mathfrak{g}^{\mathrm{hor},Y},\, \Delta \mapsto (\mathbb{I} - \frac{1}{2}YY^T)\Delta{}Y^T - Y\Delta^T(\mathbb{I} - \frac{1}{2}YY^T)
```

The elements of ``\mathfrak{g}^{\mathrm{hor},E}=:\mathfrak{g}^\mathrm{hor}``, i.e. for the special case ``Y=E``. Its elements are of the form described on top of this page.

## Special functions

You can also draw random elements from $\mathfrak{g}^\mathrm{hor}$ through e.g. 
```julia
rand(CUDADevice(), StiefelLieAlgHorMatrix{Float32}, 10, 5)
```
In this example: $N=10$ and $n=5$.