# Volume-Preserving Feedforward Layer 

The *volume-preserving feedforward layers* in `GeometricMachineLearning` are closely related to [SympNet layers](@ref "SympNet Layers"). We note that [bajars2023locally](@cite) proposes more complicated volume-preserving feedforward neural networks than the ones presented here. These are motivated by a classical theorem [kang1995volume](@cite). But for reasons of simplicity and their similarity to SympNet layers we resort to the ones discussed below.

[Volume-preserving](@ref "Divergence-Free Vector Fields") feedforward layers in `GeometricMachineLearning` are a special type of ResNet layer for which we restrict the weight matrices to be of a particular form. Each layer computes: 

```math
\mathtt{VPFF}_{A, b}: x \mapsto x + \sigma(Ax + b),
```
where ``\sigma`` is a nonlinearity, ``A`` is the weight and ``b`` is the bias. The matrix ``A`` is either a [`LowerTriangular`](@ref) matrix ``L`` or an [`UpperTriangular`](@ref) matrix[^1] ``U``. We demonstrate volume-preservation of these layers by considering the case ``A = L``. The matrix looks as follows:

[^1]: Implemented as `LowerTriangular` and `UpperTriangular` in `GeometricMachineLearning`.



```math 
L = \begin{pmatrix}
     0 & 0 & \cdots & 0      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
```

For the jacobian we then have:

```math 
J = \nabla\mathtt{VPFF}_{L, b} = \begin{pmatrix}
     1 & 0 & \cdots & 0      \\
     b_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     b_{n1} & \cdots & b_{n(n-1)}      & 1 
\end{pmatrix},
```
and the determinant of ``J`` is 1, i.e. the map is volume-preserving. A similar statement holds if the matrix ``A`` is [`UpperTriangular`](@ref) instead of [`LowerTriangular`](@ref).

## Library Functions 

```@docs
VolumePreservingFeedForwardLayer
VolumePreservingLowerLayer
VolumePreservingUpperLayer
```