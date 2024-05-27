# Symmetric Skew-Symmetric and Triangular Matrices.

Among the special arrays implemented in `GeometricMachineLearning` [`SymmetricMatrix`](@ref), [`SkewSymMatrix`](@ref), [`UpperTriangular`](@ref) and [`LowerTriangular`](@ref) are the most common ones and these can also be found in other libraries; `LinearAlgebra.jl` has an implementation of a symmetric matrix called `Symmetric` for example. The versions of these matrices in `GeometricMachineLearning` are however more memory efficient as they only store as many parameters as are necessary, i.e. ``n(n+1)/2`` for the symmetric matrix and ``n(n-1)/2`` for the other three. In addition operations such as matrix and tensor multiplication are implemented for these matrices to work in parallel on GPU. 

We now show the various matrices. First [`UpperTriangular`](@ref):

```math 
U = \begin{pmatrix}
     0 & a_{12} & \cdots & a_{1n}      \\
     0 & \ddots &        & a_{2n} \\
     \vdots & \ddots & \ddots & \vdots \\
     0 & \cdots & 0      & 0 
\end{pmatrix}.
```

The matrix [`LowerTriangular`](@ref):

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
where the first part of this matrix is skew-symmetric and the second part is symmetric. This is also how the constructors for [`SkewSymMatrix`](@ref) and [`SymmetricMatrix`](@ref) are designed:

```@example sym_skew_sym_example
using GeometricMachineLearning

M = rand(3, 3) 
```

```@example sym_skew_sym_example
A = SkewSymMatrix(M)
```

```@example sym_skew_sym_example
B = SymmetricMatrix(M)
```

```@example sym_skew_sym_example
@assert M  ≈ A + B # hide
M  ≈ A + B
```

## How are Special Matrices Stored?

The following image demonstrates how special matrices are stored in `GeometricMachineLearning`:

```@example 
Main.include_graphics("../tikz/skew_sym_visualization"; caption = "The elements of a skew-symmetric matrix (and other special matrices) are stored as a vector. The elements of the big vector are the entries on the lower left of the matrix, stored row-wise.") # hide
```

So what is stored internally is a vector of size ``n(n-1)/2`` for the skew-symmetric matrix and the triangular matrices and a vector of size ``n(n+1)/2`` for the symmetric matrix. We can sample a random skew-symmetric matrix: 

```@example skew_sym
using GeometricMachineLearning 
import Random 
Random.seed!(123)

A = rand(SkewSymMatrix, 5)
```

and then access the vector:

```@example skew_sym
A.S 
```

This is equivalent to sampling a vector and then assigning a matrix:

```@example skew_sym
using GeometricMachineLearning
import Random
Random.seed!(123)

S = rand(5 * (5 - 1) ÷ 2)
SkewSymMatrix(S, 5)
```

These special matrices are important for [SympNets](@ref "SympNet Architecture"), [volume-preserving transformers](@ref "Volume-Preserving Transformer") and [linear symplectic transformers](@ref "Linear Symplectic Transformer").

## Parallel Computation

The functions [`GeometricMachineLearning.mat_tensor_mul`](@ref) and [`GeometricMachineLearning.tensor_mat_mul`](@ref) are also implemented for these matrices for efficient parallel computations. This is elaborated on when we introduce [pullbacks](@ref "Pullbacks and Automatic Differentiation").

## Library Functions

```@docs; canonical = false
UpperTriangular
UpperTriangular(::AbstractMatrix)
LowerTriangular
LowerTriangular(::AbstractMatrix)
SymmetricMatrix
SymmetricMatrix(::AbstractMatrix)
SkewSymMatrix
SkewSymMatrix(::AbstractMatrix)
```