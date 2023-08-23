using LinearAlgebra
using Test

using GeometricMachineLearning: BlockIdentityLowerMatrix, SymmetricBlockIdentityLowerMatrix
using GeometricMachineLearning: BlockIdentityUpperMatrix, SymmetricBlockIdentityUpperMatrix
using GeometricMachineLearning: ZeroVector


W = rand(2,2)
S = W .+ W'
x = ones(4)
y = zero(x)

@test mul!(y, BlockIdentityLowerMatrix(W), x) == vcat(ones(2), ones(2) .+ W * ones(2))
@test mul!(y, BlockIdentityUpperMatrix(W), x) == vcat(ones(2) .+ W * ones(2), ones(2))

@test mul!(y, SymmetricBlockIdentityLowerMatrix(W), x) == vcat(ones(2), ones(2) .+ S * ones(2))
@test mul!(y, SymmetricBlockIdentityUpperMatrix(W), x) == vcat(ones(2) .+ S * ones(2), ones(2))


z = ZeroVector(Float64, 4)

@test z[1] == zero(Float64)
@test z[4] == zero(Float64)

@test_throws AssertionError z[0]
@test_throws AssertionError z[5]
