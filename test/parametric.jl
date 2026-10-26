using GeometricMachineLearning
using GeometricMachineLearning: concatenate_array_with_parameters, apply_parametric,
                                NullParameters, ParametricLoss, ZygotePullback
using FiniteDifferences: central_fdm, grad
using LinearAlgebra: norm
using Test
import Random
import Zygote

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

    # one `μ` below every column of every slice of a 3-tensor
    tensor = reshape(T[1:12;], 2, 2, 3)
    μ2_flat = T[2, 1, -3]                     # the `flat` of `μ(2)`, written out
    concatenated = concatenate_array_with_parameters(tensor, μ(2))
    @test size(concatenated) == (2 + length(μ2_flat), 2, 3)
    @test eltype(concatenated) === T
    for (k, j) in Iterators.product(1:3, 1:2)
        @test concatenated[:, j, k] == vcat(tensor[:, j, k], μ2_flat)
    end

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

# The one-sample method takes a vector or a `(q, p)` pair of vectors; a pair of tensors with one `μ`
# matches no method.
@testset "apply_parametric takes one sample as vectors" begin
    ps = NetworkParameters((L1 = (W = [0.5 -1 0.25; 1.5 0.75 -2], b = [0.1, -0.3]),))
    model = Chain(ParametricAffine{2}(1))
    X = reshape([0.3, -0.7, 1.1, 0.2], 2, 1, 2)
    @test applicable(apply_parametric, model, (q = [0.3], p = [-0.7]), (a = 2.0,), ps)
    @test !applicable(
        apply_parametric, model, (q = X[1:1, :, :], p = X[2:2, :, :]), (a = 2.0,), ps)
end

# The bytes that the pullback of `ParametricLoss` and the call of its closure allocate on `data`: the
# least of three readings after two calls that compile.
function pullback_bytes(model, ps, data)
    function run()
        value, back = ZygotePullback(ParametricLoss())(ps, model, data)
        back(one(value))
    end
    run()
    run()
    minimum(@allocated(run()) for _ in 1:3)
end

# A batch of `k` samples costs O(k): the bytes of one more sample, between 64 and 1024 samples, are
# at most twice the bytes per sample at 64. A sample indexed one at a time in the differentiated path
# costs a cotangent of the whole batch, O(k²) in all.
@testset "the pullback of a batch allocates linearly in its size ($form)" for form in (:array, :qp)
    ps = NetworkParameters((L1 = (W = [0.5 -1 0.25; 1.5 0.75 -2], b = [0.1, -0.3]),))
    model = Chain(ParametricAffine{2}(1))
    function batch(k)
        rng = Random.Xoshiro(k)
        X, Y = randn(rng, 2, 1, k), randn(rng, 2, 1, k)
        μs = [(a = randn(rng),) for _ in 1:k]
        form === :array ? (X, Y, μs) :
        ((q = X[1:1, :, :], p = X[2:2, :, :]), (q = Y[1:1, :, :], p = Y[2:2, :, :]), μs)
    end
    b64 = pullback_bytes(model, ps, batch(64))
    b1024 = pullback_bytes(model, ps, batch(1024))
    @test (b1024 - b64) / 960 ≤ 2 * b64 / 64
end

# The gradient of the loss with respect to the input, against central finite differences of the same
# loss over the entries of the input.
@testset "the input gradient of ParametricLoss ($form)" for form in (:array, :qp)
    ps = NetworkParameters((L1 = (W = [0.5 -1 0.25; 1.5 0.75 -2], b = [0.1, -0.3]),))
    model = Chain(ParametricAffine{2}(1))
    rng = Random.Xoshiro(3)
    k = 4
    X, Y = randn(rng, 2, 1, k), randn(rng, 2, 1, k)
    μs = [(a = randn(rng),) for _ in 1:k]
    as_input(v) = form === :array ? reshape(v, 2, 1, k) :
                  (q = reshape(v[1:k], 1, 1, k), p = reshape(v[(k + 1):end], 1, 1, k))
    as_vector(x) = form === :array ? vec(x) : vcat(vec(x.q), vec(x.p))
    y = form === :array ? Y : (q = Y[1:1, :, :], p = Y[2:2, :, :])
    loss(x) = ParametricLoss()(model, ps, x, y, μs)
    v = form === :array ? vec(X) : vcat(vec(X[1, 1, :]), vec(X[2, 1, :]))

    computed = only(Zygote.gradient(loss, as_input(v)))
    @test computed !== nothing
    reference = only(grad(central_fdm(5, 1), loss ∘ as_input, v))
    @test norm(as_vector(computed) - reference) ≤ 1e-6 * norm(reference)
end

# The gradient that the pullback of `ParametricLoss` returns infers as its run-time type, for one sample
# and for a batch. The `(q, p)` form is left out: AbstractNeuralNetworks' loss of a `(q, p)` named tuple
# infers its gradient as a `Union` on its own (KNOWN_ISSUES K30).
@testset "the gradient of ParametricLoss infers concretely ($T)" for T in REAL_ELTYPES
    ps = NetworkParameters((L1 = (W = T[0.5 -1 0.25; 1.5 0.75 -2], b = T[0.1, -0.3]),))
    model = Chain(ParametricAffine{2}(1))
    rng = Random.Xoshiro(5)
    k = 4
    X, Y = randn(rng, T, 2, 1, k), randn(rng, T, 2, 1, k)
    μs = [(a = randn(rng, T),) for _ in 1:k]
    for data in ((X[:, 1, 1], Y[:, 1, 1], μs[1]), (X, Y, μs))
        value, back = ZygotePullback(ParametricLoss())(ps, model, data)
        @test (@inferred back(one(value))) isa NetworkParameters{T}
    end
end

@testset "the parametric names are public and not exported" begin
    for name in (:apply_parametric, :ParametricLoss, :ParametricDataLoader)
        @test Base.ispublic(GeometricMachineLearning, name)
        @test !Base.isexported(GeometricMachineLearning, name)
    end
end
