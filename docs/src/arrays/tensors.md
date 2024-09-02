# Tensors in `GeometricMachineLearning`

We typically store training data as *tensors with three axes* in `GeometricMachineLearning`. This allows for a parallel computation of matrix products, also for the special arrays such as [`LowerTriangular`](@ref), [`UpperTriangular`](@ref), [`SymmetricMatrix`](@ref) and [`SkewSymMatrix`](@ref) and objects of [`Manifold`](@ref) type such as the [`StiefelManifold`](@ref). 

## Library Functions

```@docs
GeometricMachineLearning.tensor_mat_mul(::AbstractArray{<:Number, 3}, ::AbstractMatrix)
GeometricMachineLearning.tensor_mat_mul!(::AbstractArray{<:Number, 3}, ::AbstractArray{<:Number, 3}, ::AbstractMatrix)
GeometricMachineLearning.tensor_mat_mul!(::AbstractArray{<:Number, 3}, ::AbstractArray{<:Number, 3}, ::SymmetricMatrix)
GeometricMachineLearning.mat_tensor_mul(::AbstractMatrix, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::AbstractMatrix, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::LowerTriangular, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::UpperTriangular, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::SkewSymMatrix, ::AbstractArray{<:Number, 3})
GeometricMachineLearning.mat_tensor_mul!(::AbstractArray{<:Number, 3}, ::SymmetricMatrix, ::AbstractArray{<:Number, 3})
```