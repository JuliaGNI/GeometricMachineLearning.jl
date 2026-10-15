using GeometricMachineLearning
using GeometricMachineLearning: _mul
using LinearAlgebra: Adjoint
using Zygote: gradient
using Test
import Random

Random.seed!(123)

const STRUCTURED = (
    SymmetricMatrix, SkewSymMatrix, StrictlyLowerTriangular, StrictlyUpperTriangular)

# `_mul(W, x)[:, :, k] == W * x[:, :, k]`, slice by slice, with the structured weight multiplied as a
# dense matrix.
function slices(f, x::AbstractArray{<:Number, 3})
    cat((f(x[:, :, k]) for k in axes(x, 3))...; dims = 3)
end

@testset "_mul: left, right and tensor products against slice-by-slice references" begin
    for T in (Float32, Float64)
        x = rand(T, 4, 5, 3)
        for W in (rand(T, 6, 4), rand(T, 4, 6)', (rand(TT{T}, 4) for TT in STRUCTURED)...)
            Wd = Matrix(W)
            y = @inferred _mul(W, x)
            @test y isa Array{T, 3}
            @test y ≈ slices(xk -> Wd * xk, x)
            @test _mul(W, x[:, :, 1]) ≈ Wd * x[:, :, 1]
            @test _mul(W, x[:, 1, 1]) ≈ Wd * x[:, 1, 1]
            @test _mul(W, x[:, :, 1]) isa Matrix{T}
            @test _mul(W, x[:, 1, 1]) isa Vector{T}
        end
        z = rand(T, 3, 4, 5)
        for W in (rand(T, 4, 6), (rand(TT{T}, 4) for TT in STRUCTURED)...)
            y = @inferred _mul(z, W)
            @test y isa Array{T, 3}
            @test y ≈ slices(zk -> zk * Matrix(W), z)
        end
        y = rand(T, 5, 2, 3)
        @test @inferred(_mul(x, y)) isa Array{T, 3}
        @test _mul(x, y) ≈ cat((x[:, :, k] * y[:, :, k] for k in 1:3)...; dims = 3)
    end
end

# A matrix argument on the right of a left product is multiplied as it stands, not through a
# reshape, which would turn a structured matrix into a `ReshapedArray`.
@testset "_mul: a matrix times a structured matrix" begin
    for T in (Float32, Float64), TT in STRUCTURED

        z = rand(T, 3, 4)
        A = rand(TT{T}, 4)
        @test _mul(z, A) isa Matrix{T}
        @test _mul(z, A) ≈ z * Matrix(A)
    end
end

# GEMM promotes, so a weight and data of two precisions multiply; no element type is checked.
@testset "_mul: mixed element types promote" begin
    W = rand(Float32, 3, 4)
    x = rand(Float64, 4, 2, 5)
    @test _mul(W, x) isa Array{Float64, 3}
    @test _mul(Float64.(W), Float32.(x)) isa Array{Float64, 3}
    @test _mul(W', rand(Float64, 3, 2, 5)) isa Array{Float64, 3}
end

# The cotangent of a dense `Matrix` parameter multiplied as `W'` is a `Matrix`, not an `Adjoint`,
# at every rank of the data.
@testset "_mul: the cotangent of W in _mul(W', x) is a Matrix" begin
    for T in (Float32, Float64)
        W = rand(T, 4, 3)
        for x in (rand(T, 4), rand(T, 4, 5), rand(T, 4, 5, 2))
            g = gradient(w -> sum(abs2, _mul(w', x)), W)[1]
            reference = gradient(w -> sum(abs2, Matrix(w') * reshape(x, 4, :)), W)[1]
            @test g isa Matrix{T}
            @test isapprox(g, reference; rtol = sqrt(eps(T)))
        end
    end
end
