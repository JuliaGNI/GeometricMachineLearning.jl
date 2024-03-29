# The Cayley Retraction 

The Cayley transformation is one of the most popular retractions. For several matrix Lie groups it is a mapping from the Lie algebra $\mathfrak{g}$ onto the Lie group $G$. 
They Cayley retraction reads: 

```math
    \mathrm{Cayley}(C) = \left(\mathbb{I} -\frac{1}{2}C\right)^{-1}\left(\mathbb{I} +\frac{1}{2}C\right).
```
This is easily checked to be a retraction, i.e. $\mathrm{Cayley}(\mathbb{O}) = \mathbb{I}$ and $\frac{\partial}{\partial{}t}\mathrm{Cayley}(tC) = C$.

What we need in practice is not the computation of the Cayley transform of an arbitrary matrix, but the Cayley transform of an element of $\mathfrak{g}^\mathrm{hor}$, the [global tangent space representation](../../arrays/stiefel_lie_alg_horizontal.md). 

The elements of $\mathfrak{g}^\mathrm{hor}$ can be written as: 

```math
C = \begin{bmatrix}
    A & -B^T \\ 
    B & \mathbb{O}
\end{bmatrix} = \begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix},
```

where the second expression exploits the sparse structure of the array, i.e. it is a multiplication of a $N\times2n$ with a $2n\times{}N$ matrix. We can hence use the **Sherman-Morrison-Woodbury formula** to obtain:

```math
(\mathbb{I} - \frac{1}{2}UV)^{-1} = \mathbb{I} + \frac{1}{2}U(\mathbb{I} - \frac{1}{2}VU)^{-1}V
```

So what we have to invert is the term 

```math
\mathbb{I} - \frac{1}{2}\begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix}\begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} = 
\begin{bmatrix}  \mathbb{I} - \frac{1}{4}A & - \frac{1}{2}\mathbb{I} \\ \frac{1}{2}B^TB - \frac{1}{8}A^2 & \mathbb{I} - \frac{1}{4}A  \end{bmatrix}.
```

The whole Cayley transform is then: 

```math
\left(\mathbb{I} + \frac{1}{2}\begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} - \frac{1}{4}A & - \frac{1}{2}\mathbb{I} \\ \frac{1}{2}B^TB - \frac{1}{8}A^2 & \mathbb{I} - \frac{1}{4}A  \end{bmatrix}^{-1}  \begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix} \right)\left( E +  \frac{1}{2}\begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} \\ \frac{1}{2}A   \end{bmatrix}\ \right) = \\
E + \frac{1}{2}\begin{bmatrix} \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O}  \end{bmatrix}\left(
    \begin{bmatrix}  \mathbb{I} \\ \frac{1}{2}A   \end{bmatrix}  + 
    \begin{bmatrix}  \mathbb{I} - \frac{1}{4}A & - \frac{1}{2}\mathbb{I} \\ \frac{1}{2}B^TB - \frac{1}{8}A^2 & \mathbb{I} - \frac{1}{4}A  \end{bmatrix}^{-1}\left(
        \begin{bmatrix}  \mathbb{I} \\ \frac{1}{2}A   \end{bmatrix} + 
        \begin{bmatrix} \frac{1}{2}A \\ \frac{1}{4}A^2 - \frac{1}{2}B^TB \end{bmatrix}
    \right)
    \right)
```


Note that for computational reason we compute $\mathrm{Cayley}(C)E$ instead of just the Cayley transform (see the section on [retractions](retractions.md)).