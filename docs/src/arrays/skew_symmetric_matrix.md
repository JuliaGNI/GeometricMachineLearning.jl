# Symmetric Skew-Symmetric and Triangular Matrices.

Among the special arrays implemented in `GeometricMachineLearning` [`SymmetricMatrix`](@ref), [`SkewSymMatrix`](@ref), [`UpperTriangular`](@ref) and [`LowerTriangular`](@ref) are the most common ones and these can also be found in other libraries; `LinearAlgebra.jl` has an implementation of a symmetric matrix called [`Symmetric`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Symmetric) for example. The versions of these matrices in `GeometricMachineLearning` are however more memory efficient as they only store as many parameters as are necessary, i.e. ``n(n+1)/2`` for the symmetric matrix and ``n(n-1)/2`` for the other three. In addition operations such as matrix and tensor multiplication are implemented for these matrices to work in parallel on GPU via [`GeometricMachineLearning.tensor_mat_mul`](@ref) for example. We here give an overview of *elementary* custom matrices that are implemented in `GeometricMachineLearning`. More *involved* matrices are the so-called [global tangent spaces](@ref "Global Tangent Spaces").

## Custom Matrices

`GeometricMachineLearning` has two types of *triangular matrices*. The first one is [`UpperTriangular`](@ref):

```math 
U = \begin{pmatrix}
     0 & a_{12} & \cdots & a_{1n}      \\
     0 & \ddots &        & a_{2n} \\
     \vdots & \ddots & \ddots & \vdots \\
     0 & \cdots & 0      & 0 
\end{pmatrix}.
```

And the second one is [`LowerTriangular`](@ref):

```math 
L = \begin{pmatrix}
     0 & 0 & \cdots & 0      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
```

An instance of [`SkewSymMatrix`](@ref) can be written as ``A = L - L^T`` or ``A = U - U^T``:

```math 
A = \begin{pmatrix}
     0 & - a_{21} & \cdots & - a_{n1}     \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
```

And lastly a [`SymmetricMatrix`](@ref):

```math 
L = \begin{pmatrix}
     a_{11} & a_{21} & \cdots & a_{n1}      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & a_{nn}
\end{pmatrix}.
```

Note that any matrix ``M\in\mathbb{R}^{n\times{}n}`` can be written

```math
M = \frac{1}{2}(M - M^T) + \frac{1}{2}(M + M^T),
```
where the first part of this matrix is skew-symmetric and the second part is symmetric. This is also how the constructors for [`SkewSymMatrix`](@ref) and [`SymmetricMatrix`](@ref) are designed. Consider an arbitrary matrix:

```@example sym_skew_sym_example
using GeometricMachineLearning  # hide

M = [1; 2; 3;; 4; 5; 6;; 7; 8; 9]
```

Calling [`SkewSymMatrix`](@ref) on ``M`` is equivalent to doing ``M \to \frac{1}{2}(M - M^T)``:

```@example sym_skew_sym_example
A = SkewSymMatrix(M)
```

And calling [`SymmetricMatrix`](@ref) on ``M`` is equivalent to doing ``M \to \frac{1}{2}(M + M^T)``:

```@example sym_skew_sym_example
B = SymmetricMatrix(M)
```

We can further confirm the identity above:

```@example sym_skew_sym_example
@assert M  ≈ A + B # hide
M  ≈ A + B
```

## How are Special Matrices Stored?

The following image demonstrates how a skew-symmetric matrix is stored in `GeometricMachineLearning`:

```@example 
Main.include_graphics("../tikz/skew_sym_visualization"; caption = "The elements of a skew-symmetric matrix (and other special matrices) are stored as a vector. The elements of the big vector are the entries on the lower left of the matrix, stored row-wise.") # hide
```

So what is stored internally is a vector of size ``n(n-1)/2`` for the skew-symmetric matrix and the triangular matrices and a vector of size ``n(n+1)/2`` for the symmetric matrix. We can sample a random skew-symmetric matrix: 

```@example skew_sym
using GeometricMachineLearning # hide
import Random # hide
Random.seed!(123) # hide

A = rand(SkewSymMatrix, 3)
```

and then access the vector:

```@example skew_sym
A.S 
```

This is equivalent to sampling a vector and then assigning a matrix[^1]:

[^1]: We fixed the seed to the same value in both these examples.

```@example skew_sym
using GeometricMachineLearning # hide
import Random # hide
Random.seed!(123) # hide

S = rand(3 * (3 - 1) ÷ 2)
SkewSymMatrix(S, 3)
```

These special matrices are important for [SympNets](@ref "SympNet Architecture"), [volume-preserving transformers](@ref "Volume-Preserving Transformer") and [linear symplectic transformers](@ref "Linear Symplectic Transformer").

## Parallel Computation

The functions [`GeometricMachineLearning.mat_tensor_mul`](@ref) and [`GeometricMachineLearning.tensor_mat_mul`](@ref) are also implemented for these matrices for efficient parallel computations. This is elaborated on when we take about [tensors](@ref "Tensors in `GeometricMachineLearning`").

## Library Functions

```@docs
GeometricMachineLearning.AbstractTriangular
UpperTriangular
UpperTriangular(::AbstractMatrix)
LowerTriangular
LowerTriangular(::AbstractMatrix)
vec(::GeometricMachineLearning.AbstractTriangular)
SymmetricMatrix
SymmetricMatrix(::AbstractMatrix)
SkewSymMatrix
SkewSymMatrix(::AbstractMatrix)
vec(::SkewSymMatrix)
```