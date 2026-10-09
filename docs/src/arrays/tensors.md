# Tensors in `GeometricMachineLearning`

We typically store training data as *tensors with three axes* in `GeometricMachineLearning`. This allows for a parallel computation of matrix products, also for the special arrays such as [`StrictlyLowerTriangular`](@extref GeometricOptimizers GeometricOptimizers.StrictlyLowerTriangular), [`StrictlyUpperTriangular`](@extref GeometricOptimizers GeometricOptimizers.StrictlyUpperTriangular), [`SymmetricMatrix`](@extref GeometricOptimizers GeometricOptimizers.SymmetricMatrix) and [`SkewSymMatrix`](@extref GeometricOptimizers GeometricOptimizers.SkewSymMatrix) and objects of [`Manifold`](@extref GeometricOptimizers GeometricOptimizers.Manifold) type such as the [`StiefelManifold`](@extref GeometricOptimizers GeometricOptimizers.StiefelManifold). 

## Library Functions

```@docs
GeometricMachineLearning.tensor_mat_mul(::AbstractArray{<:Number, 3}, ::AbstractMatrix)
GeometricMachineLearning.tensor_mat_mul!(::AbstractArray{<:Number, 3}, ::AbstractArray{<:Number, 3}, ::AbstractMatrix)
GeometricMachineLearning.tensor_mat_mul!(::AbstractArray{<:Number, 3}, ::AbstractArray{<:Number, 3}, ::SymmetricMatrix)
GeometricMachineLearning.mat_tensor_mul(::AbstractMatrix, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::AbstractMatrix, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::StrictlyLowerTriangular, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::StrictlyUpperTriangular, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::SkewSymMatrix, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::SymmetricMatrix, ::AbstractArray{<:Number, 3})
```