# `SymmetricMatrix` and `SkewSymMatrix`

There are special implementations of symmetric and skew-symmetric matrices in `GeometricMachineLearning.jl`. They are implemented to work on GPU and for multiplication with tensors. The following image demonstrates how the data necessary for an instance of `SkewSymMatrix` are stored[^1]:

[^1]: It works similarly for `SymmetricMatrix`. 

```@example 
Main.include_graphics("../tikz/skew_sym_visualization")
```

So what is stored internally is a vector of size ``n(n-1)/2`` for the skew-symmetric matrix and a vector of size ``n(n+1)/2`` for the symmetric matrix. We can sample a random skew-symmetric matrix: 

```@example skew_sym
using GeometricMachineLearning # hide 

A = rand(SkewSymMatrix, 5)
```

and then access the vector:

```@example skew_sym
A.S 
```