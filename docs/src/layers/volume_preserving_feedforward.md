# Volume-Preserving Feedforward Layer 

**Volume preserving feedforward layers** are a special type of ResNet layer for which we restrict the weight matrices to be of a particular form. I.e. each layer computes: 

```math
x \mapsto x + \sigma(Ax + b),
```
where ``\sigma`` is a nonlinearity, ``A`` is the weight and ``b`` is the bias. The matrix ``A`` is either a lower-triangular matrix ``L`` or an upper-triangular matrix ``U``[^1]. The lower triangular matrix is of the form (the upper-triangular layer is simply the transpose of the lower triangular): 

[^1]: Implemented as `LowerTriangular` and `UpperTriangular` in `GeometricMachineLearning`.

```math 
L = \begin{pmatrix}
     0 & 0 & \cdots & 0      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
```

The Jacobian of a layer of the above form then is of the form

```math 
J = \begin{pmatrix}
     1 & 0 & \cdots & 0      \\
     b_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     b_{n1} & \cdots & b_{n(n-1)}      & 1 
\end{pmatrix},
```
and the determinant of ``J`` is 1, i.e. the map is volume-preserving. 

## Library Functions 

```@docs; canonical=false
VolumePreservingFeedForwardLayer
```