# Tensors in `GeometricMachineLearning`

We typically store training data as *tensors with three axes* in `GeometricMachineLearning`. This allows for a parallel computation of matrix products, also for the special arrays such as [`StrictlyLowerTriangular`](@extref GeometricOptimizers GeometricOptimizers.StrictlyLowerTriangular), [`StrictlyUpperTriangular`](@extref GeometricOptimizers GeometricOptimizers.StrictlyUpperTriangular), [`SymmetricMatrix`](@extref GeometricOptimizers GeometricOptimizers.SymmetricMatrix) and [`SkewSymMatrix`](@extref GeometricOptimizers GeometricOptimizers.SkewSymMatrix) and objects of [`Manifold`](@extref GeometricOptimizers GeometricOptimizers.Manifold) type such as the [`StiefelManifold`](@extref GeometricOptimizers GeometricOptimizers.StiefelManifold). 

## Library Functions

```@docs
GeometricMachineLearning._mul
```
