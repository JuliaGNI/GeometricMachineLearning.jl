# Horizontal component of the $\mathfrak{g}$

What we use to optimize Adam (and other algorithms) to manifolds is a **global tangent space representation** of the homogeneous spaces. 

For the Stiefel manifold, the homogeneous space takes a simple form: 
```math 
B = \begin{bmatrx}
    A & -B^T \\ 
    B & \mathbb{O}
\end{bmatrix}.
```

## Theoretical foundations of global tangent space representation

### Vertical and horizontal components

The Stiefel manifold is a homogeneous space obtained from $SO(N)$ by setting two matrices whose first $n$ columns conincide equivalent. 
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