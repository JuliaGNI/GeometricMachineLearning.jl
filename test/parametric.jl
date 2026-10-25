using GeometricMachineLearning
using GeometricMachineLearning: concatenate_array_with_parameters, apply_parametric,
                                NullParameters
using Test
import Random

include("helpers/eltypes.jl")
include("helpers/parametric_affine.jl")

# System parameters in `Float64`, with a vector-valued leaf; `flat` writes out by hand what
# flattening one of them gives in the element type `T`: the leaves in the order of the keys.
μ(j) = (a = Float64(j), b = [0.5j, -1.5j])
flat(::Type{T}, μ) where {T} = T[μ.a; μ.b[1]; μ.b[2]]

@testset "concatenate_array_with_parameters ($T)" for T in REAL_ELTYPES
    x = T[1 2 3; 4 5 6]

    # one `μ` per column: the columns side by side
    concatenated = concatenate_array_with_parameters(x, [μ(1), μ(2), μ(3)])
    @test concatenated == T[1 2 3; 4 5 6; 1 2 3; 0.5 1 1.5; -1.5 -3 -4.5]
    @test eltype(concatenated) === T

    # one `μ` below a vector
    concatenated = concatenate_array_with_parameters(x[:, 2], μ(2))
    @test concatenated == T[2, 5, 2, 1, -3]
    @test eltype(concatenated) === T

    # one `μ` below every column of a matrix
    concatenated = concatenate_array_with_parameters(x, μ(2))
    @test concatenated == T[1 2 3; 4 5 6; 2 2 2; 1 1 1; -3 -3 -3]
    @test eltype(concatenated) === T

    # no system parameters: the state itself
    @test concatenate_array_with_parameters(x, NullParameters()) === x
    @test concatenate_array_with_parameters(x[:, 1], NullParameters()) == x[:, 1]
    # an empty named tuple concatenates nothing
    @test concatenate_array_with_parameters(x[:, 1], (;)) == x[:, 1]
    @test eltype(concatenate_array_with_parameters(x[:, 1], (;))) === T

    @test_throws DimensionMismatch concatenate_array_with_parameters(x, [μ(1), μ(2)])
end

# `ParametricAffine` maps `(x, μ)` to `W [x; flatten(μ)] + b`; `closed_form` writes that out with the
# flattened `μ` built by hand.
@testset "apply_parametric against the closed form ($T)" for T in REAL_ELTYPES
    W = T[0.5 -1 0.25 2 -0.5; 1.5 0.75 -2 0.125 1]
    b = T[0.1, -0.3]
    ps = NetworkParameters((L1 = (W = W, b = b),))
    model = Chain(ParametricAffine{2}(3))
    closed_form(x, μ) = W * vcat(x, flat(T, μ)) + b
    tolerance = 10 * eps(T)

    # one sample: a vector, and a `(q, p)` pair of vectors
    x = T[0.3, -0.7]
    output = apply_parametric(model, x, μ(2), ps)
    @test size(output) == size(x)
    @test eltype(output) === T
    @test isapprox(output, closed_form(x, μ(2)); rtol = tolerance)

    output = apply_parametric(model, (q = x[1:1], p = x[2:2]), μ(2), ps)
    @test size(output.q) == size(output.p) == (1,)
    @test eltype(output.q) === eltype(output.p) === T
    @test isapprox(vcat(output.q, output.p), closed_form(x, μ(2)); rtol = tolerance)

    # a batch of `k` samples with `k` values of `μ`: a `2d × 1 × k` tensor, and a `(q, p)` pair of
    # `d × 1 × k` tensors
    k = 4
    μs = [μ(j) for j in 1:k]
    X = reshape(T[0.3, -0.7, 1.1, 0.2, -0.4, 0.9, 0.0, 2.5], 2, 1, k)
    expected = reshape(reduce(hcat, [closed_form(X[:, 1, i], μs[i]) for i in 1:k]), 2, 1, k)

    output = apply_parametric(model, X, μs, ps)
    @test size(output) == size(X)
    @test eltype(output) === T
    @test isapprox(output, expected; rtol = tolerance)

    output = apply_parametric(model, (q = X[1:1, :, :], p = X[2:2, :, :]), μs, ps)
    @test size(output.q) == size(output.p) == (1, 1, k)
    @test eltype(output.q) === eltype(output.p) === T
    @test isapprox(output.q, expected[1:1, :, :]; rtol = tolerance)
    @test isapprox(output.p, expected[2:2, :, :]; rtol = tolerance)

    # one `μ` per sample and one time step per sample
    @test_throws DimensionMismatch apply_parametric(model, X, μs[1:3], ps)
    @test_throws DimensionMismatch apply_parametric(
        model, reshape(X, 2, 2, 2), μs[1:2], ps)
end

@testset "the parametric names are public and not exported" begin
    for name in (:apply_parametric, :ParametricLoss, :ParametricDataLoader)
        @test Base.ispublic(GeometricMachineLearning, name)
        @test !Base.isexported(GeometricMachineLearning, name)
    end
end
