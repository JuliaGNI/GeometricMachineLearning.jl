# Stiefel manifold 

The Stiefel manifold $St(n, N)$ is the space of all orhtonormal frames in $\mathbb{R}^{N\times{}n}$, i.e. matrices $Y\in\mathbb{R}^{N\times{}n}$ s.t. $Y^TY = I$. It can also be seen as the orthonormal group $O(N)$ modulo an equivalence relation: $A\sim{}B\iff\exists{}C\text{ s.t. }AC = B$ for 

$$
C = \begin{pmatrix}
    I & 0 \\
    0 & Q 
\end{pmatrix}
$$

and $Q\in{}O(N-n)$, so the first $n$ columns of $A$ and $B$ are equivalent.

The tangent space to the element $Y\in{}St(n,N)$ can easily be determined: $T_YSt(n,N)=\{\Delta:\Delta^TY + Y^T\Delta = 0\}$. $St(n, N)$ is furthermore a **homogeneous space** because $O(N)$ acts transitively on it, hence its tangent space can be described through $\mathfrak{g}$, the Lie algebra of $O(N)$ via $T_YSt(n,N) = \mathfrak{g}\cdot{}Y$. Based on the element $Y$, $\mathfrak{g}$ can be split into a vertical and a horizontal component: $\mathfrak{g} = \mathfrak{g}^{\mathrm{ver},Y}\oplus\mathfrak{g}^{\mathrm{hor},Y}$, with $\mathfrak{g}^{\mathrm{ver},Y} := \{V\in\mathfrak{g}:VY = 0\}$ and the horizontal component is computed according to the canonical metric on $\mathfrak{g}$, i.e. is the orthogonal complement to $\mathfrak{g}^{\mathrm{ver},Y}$.

The function `rgrad` is a mapping that takes an element of $St(n,N)$ and a "Euclidean gradient" and produces an element $\in{}T_YSt(n,N)$. This mapping has the property: $\mathrm{Tr}((\nabla{}f)^T\Delta) = g_Y(\mathtt{rgrad}(Y, \nabla{}f), \Delta)$ $\forall\Delta\in{}T_YSt(n,N)$ and $g$ is the Riemannian metric.

What we use for optimizing on the Stiefel manifold (especially regarding the generalization of the Adam optimizer) is the tangent space to $E:=[e_1,\ldots,e_n]$. This consists of elements: 

$$
T_ESt(n,N) = \left\{\begin{pmatrix} A \\ B \end{pmatrix}: A\text{ skew-sym. and $B$ arbitrary}\right\}.
$$

Further: 

$$
\mathfrak{g}^\mathrm{hor} = \mathfrak{g}^{\mathrm{hor},E} = \left\{\begin{pmatrix} A & -B^T \\ B & 0 \end{pmatrix}: A\text{ skew-sym. and $B$ arbitrary}\right\}.
$$