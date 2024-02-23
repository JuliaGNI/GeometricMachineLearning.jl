# Volume-Preserving Feedforward Layer 

**Volume preserving feedforward layers** are a special type of ResNet layer for which we restrict the weight matrices to be of a particular form. I.e. each layer computes: 

```math
x \mapsto x + \sigma(Ax + b),
```
where ``\sigma`` is a nonlinearity, ``A`` is the weight and ``b`` is the bias. The matrix ``A`` is either a lower-triangular matrix or an upper-triangular matrix[^1]. The lower triangular matrix is of the form: 

[^1]: Implemented as `LowerTriangular` and `UpperTriangular` in `GeometricMachineLearning`.

```math 
\begin{pmatrix}
     0 & 0 & \Cdots & 0      \\
     a_{21} & \Ddots &        & \Vdots \\
     \Vdots & \Ddots & \Ddots & \Vdots \\
     a_{n1} & \Cdots & a_{n(n-1)}      & 0 
\end{pmatrix}
```

The Jacobian of a layer of the above form then is of the form

```math 
J = \begin{pmatrix}
     1 & 0 & \Cdots & 0      \\
     b_{21} & \Ddots &        & \Vdots \\
     \Vdots & \Ddots & \Ddots & \Vdots \\
     b_{n1} & \Cdots & b_{n(n-1)}      & 1 
\end{pmatrix},
```
and the determinant of ``J`` is 1, i.e. the map is volume-preserving. 

## Note on Sympnets 

As [SympNets](../architectures/sympnet.md) are symplectic maps, they also conserve phase space volume and therefore form a subcategory of volume-preserving feedforward layers. 