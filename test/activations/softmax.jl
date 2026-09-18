# `MatrixSoftmax` is the default `attention_activation` of `SymplecticAttentionQ`,
# `SymplecticAttentionP` and `SymplecticTransformer`, applied to `QᵀAQ` with `A` a learned,
# unbounded weight. Unlike `VectorSoftmax` (which delegates to `NNlib.softmax`, and does subtract
# the maximum before exponentiating), both `MatrixSoftmax` methods computed `exp.(x)` directly, so
# a large-but-ordinary entry overflowed to `Inf` and the normalised result to `NaN`.

using GeometricMachineLearning
using Test

@testset "MatrixSoftmax is finite and normalised for $T" for T in (Float32, Float64)
    x = T[100 0; 0 0]
    y = MatrixSoftmax()(x)

    @test all(isfinite, y)
    @test sum(y) ≈ one(T)
    @test eltype(y) == T
end

@testset "MatrixSoftmax does not move the answer for well-scaled input, $T" for T in (
    Float32, Float64)
    x = T[1 2; 3 4]
    y = MatrixSoftmax()(x)

    # Subtracting the maximum is exact in exact arithmetic, so for input that does not overflow
    # either way the shifted and unshifted computations must agree, and must agree with the fixed
    # implementation's answer.
    shifted = exp.(x .- maximum(x))
    unshifted = exp.(x)
    @test shifted ./ sum(shifted) ≈ unshifted ./ sum(unshifted)
    @test y ≈ unshifted ./ sum(unshifted)
    @test eltype(y) == T
end

@testset "MatrixSoftmax 3-tensor method is finite and normalised for $T" for T in (
    Float32, Float64)
    x = cat(T[100 0; 0 0], T[1 2; 3 4]; dims = 3)
    y = MatrixSoftmax()(x)

    @test all(isfinite, y)
    @test all(≈(one(T)), sum(y, dims = (1, 2)))
    @test eltype(y) == T
end

@testset "MatrixSoftmax 3-tensor method does not move the answer for well-scaled input, $T" for T in (
    Float32, Float64)
    x = cat(T[1 2; 3 4], T[5 1; 2 3]; dims = 3)
    y = MatrixSoftmax()(x)

    shifted = exp.(x .- maximum(x, dims = (1, 2)))
    unshifted = exp.(x)
    @test shifted ./ sum(shifted, dims = (1, 2)) ≈
          unshifted ./ sum(unshifted, dims = (1, 2))
    @test y ≈ unshifted ./ sum(unshifted, dims = (1, 2))
    @test eltype(y) == T
end
