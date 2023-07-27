# Stiefel manifold 

The Stiefel manifold $St(n, N)$ is the space of all orhtonormal frames in $\mathbb{R}^{N\times{}n}$. It can also be seen as the orthonormal group $O(N)$ modulo an equivalence relation: $A~B\iff\exists{}C\text{ s.t. }AC = B$ for 
$$
C = \begin{pmatrix}
    I & 0 \\
    0 & Q 
\end{pmatrix}
$$ 
and $Q\in{}O(N-n)$, so the first $n$ columns of $A$ and $B$ are equivalent. The tangent space to the element $Y\in{}St(n,N)$ can easily be determined: $T_YSt(n,N)=\{\Delta:\Delta^TY + Y^T\Delta = 0\}$. $St(n, N)$ is furthermore a **homogeneous space** as $O(N)$ acts transitively on it, hence its tangent space can be described through $\mathfrak{g}$, the Lie algebra of $O(N)$. Based on the element $Y$, $\mathfrak{g}$ can be split into a vertical and a horizontal component: $\mathfrak{g} = \mathfrak{g}^{\mathrm{ver},Y}\oplus\mathfrak{g}^{\mathrm{hor},Y}$, with $\mathfrak{g}^{\mathrm{ver},Y} := \{V\in\mathfrak{g}:VY = 0\}$ and the horizontal component is computed according to the canonical metric on $\mathfrak{g}$.

The function `rgrad` is a mapping that takes an element of $St(n,N)$ and a "Euclidean gradient" and produces an element $\in{}T_YSt(n,N)$.

What we use for optimizing on the Stiefel manifold (especially regarding the generalization of the Adam optimizer) is the tangent space to $E:=[e_1,\ldots,e_n]$. This consists of elements: 
$$
T_ESt(n,N) = \left\{\begin{pmatrix} A \\ B \end{pmatrix}: A\text{ skew-sym. and $B$ arbitrary}\right\}.
$$

Further: 
$$
\mathfrak{g}^\mathrm{hor} = \mathfrak{g}^{\mathrm{hor},E} = \left\{\begin{pmatrix} A & -B^T \\ B & 0 \end{pmatrix}: A\text{ skew-sym. and $B$ arbitrary}\right\}.
$$