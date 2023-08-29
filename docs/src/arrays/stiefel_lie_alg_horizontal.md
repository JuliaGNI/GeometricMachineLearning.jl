# Horizontal component of the Lie algebra $\mathfrak{g}$

What we use to optimize Adam (and other algorithms) to manifolds is a **global tangent space representation** of the homogeneous spaces. 

For the Stiefel manifold, the homogeneous space takes a simple form: 
```math 
B = \begin{bmatrix}
    A & -B^T \\ 
    B & \mathbb{O}
\end{bmatrix}.
```

## Theoretical foundations of global tangent space representation

### Vertical and horizontal components

The Stiefel manifold is a homogeneous space obtained from $SO(N)$ by setting two matrices, whose first $n$ columns conincide, equivalent. 
Another way of expressing this is: 
```math
A_1 \sim A_2 \iff A_1E = A_2E
```
for 
```math 
E = \begin{bmatrix} \mathbb{I} \\ \mathbb{O}\end{bmatrix}.
```

The tangent space $T_ESt(n,N)$ can also be expressed that way:
```math
T_ESt(n,N) = \mathfrak{g}\cdot{}E = \{BE:B\in\mathfrak{g}\}.
```
The kernel of the mapping $\mathfrak{g}\to{}T_ESt(n,N), B\mapsto{}BE$ is referred to as $\mathfrak{g}^{\mathrm{ver},E}$, the **vertical component** of the Lie algebra at $E$. It is clear that elements belonging to $\mathfrak{g}^{\mathrm{ver},E}$ are of the following form: 
```math 
\begin{pmatrix}
\hat{\mathbb{O}} & \mathbb{O}^T \\ 
\mathbb{O} & C
\end{pmatrix},
```
where $\hat{\mathbb{O}}\in\mathbb{R}^{n\times{}n}$ is a "small" matrix and $\mathbb{O}\in\mathbb{R}^{N\times{}n}$ is a "big" one. $C\in\mathbb{R}^{N\times{}N}$ is a skew-symmetric matrix. 

We can then take the orthogonal complement of this matrix (with respect to the canonical metric). We will denote this by $\mathfrak{g}^{\mathrm{hor},E}\equiv\mathfrak{g}^\mathrm{hor}$. Its alements are of the form: 
```math
\begin{pmatrix}
A & -B^T \\ 
B & \tilde{\mathbb{O}}
\end{pmatrix},
```
where $A\in\mathbb{R}^{n\times{}n}$ is skew-symmetric and $B\in\mathbb{R}^{N\times{}n}$ is arbitrary. This is what the struct `StiefelLieAlgHorMatrix` implements. 

## Special functions

You can also draw random elements from $\mathfrak{g}^\mathrm{hor}$ through e.g. `rand(CUDADevice(), StiefelLieAlgHorMatrix{Float32}, 10, 5)`. Where $N=10$ and $n=5$ in this example.