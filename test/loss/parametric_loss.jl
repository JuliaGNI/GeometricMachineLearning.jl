using GeometricMachineLearning
using GeometricMachineLearning: ParametricLoss, ZygotePullback
using FiniteDifferences: central_fdm, grad
using LinearAlgebra: norm
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/parametric_affine.jl")

# System parameters in `Float64`, with a vector-valued leaf, and their flattened form written out by
# hand in the element type `T`.
μ(j) = (a = Float64(j), b = [0.5j, -1.5j])
flat(::Type{T}, μ) where {T} = T[μ.a; μ.b[1]; μ.b[2]]

const K = 4
const μS = [μ(j) for j in 1:K]

# The network `(x, μ) ↦ W [x; flatten(μ)] + b` on a batch of `K` samples, and the relative norm of its
# error, both written out with the flattened `μ` built by hand.
function closed_form(W, b, X, μs)
    T = eltype(X)
    reshape(reduce(hcat, [W * vcat(X[:, 1, i], flat(T, μs[i])) + b for i in axes(X, 3)]),
        size(X))
end
closed_form_loss(W, b, X, Y, μs) = norm(closed_form(W, b, X, μs) - Y) / norm(Y)

function data(::Type{T}) where {T}
    rng = Random.Xoshiro(7)
    W = randn(rng, T, 2, 5)
    b = randn(rng, T, 2)
    X = randn(rng, T, 2, 1, K)
    Y = randn(rng, T, 2, 1, K)
    W, b, X, Y
end

network_parameters(W, b) = NetworkParameters((L1 = (W = W, b = b),))
const MODEL = Chain(ParametricAffine{2}(3))

@testset "ParametricLoss is the relative norm of the error ($T)" for T in REAL_ELTYPES
    W, b, X, Y = data(T)
    value = ParametricLoss()(MODEL, network_parameters(W, b), X, Y, μS)
    @test value isa T
    @test isapprox(value, closed_form_loss(W, b, X, Y, μS); rtol = 10 * eps(T))
end

@testset "ZygotePullback of ParametricLoss against finite differences" begin
    W, b, X, Y = data(Float64)
    value, back = ZygotePullback(ParametricLoss())(
        network_parameters(W, b), MODEL, (X, Y, μS))
    @test isapprox(value, closed_form_loss(W, b, X, Y, μS); rtol = 10 * eps(Float64))

    gradient = back(one(value))
    # the finite-difference gradient of the closed form over the entries of `W`, then of `b`
    loss_of_entries(v) = closed_form_loss(reshape(v[1:10], 2, 5), v[11:12], X, Y, μS)
    reference = only(grad(central_fdm(5, 1), loss_of_entries, vcat(vec(W), b)))
    computed = vcat(vec(gradient.L1.W), gradient.L1.b)
    @test norm(computed - reference) ≤ 1e-6 * norm(reference)
end

# One sample with one `μ` is the other shape the pullback takes.
@testset "ZygotePullback of ParametricLoss on one sample ($T)" for T in REAL_ELTYPES
    W, b, X, Y = data(T)
    x, y = X[:, 1, 1], Y[:, 1, 1]
    value, back = ZygotePullback(ParametricLoss())(
        network_parameters(W, b), MODEL, (x, y, μ(2)))
    expected = norm(W * vcat(x, flat(T, μ(2))) + b - y) / norm(y)
    @test value isa T
    @test isapprox(value, expected; rtol = 10 * eps(T))
    @test eltype(back(one(value)).L1.W) === T
end

@testset "ZygotePullback of ParametricLoss keeps Float32" begin
    W, b, X, Y = data(Float32)
    value, back = ZygotePullback(ParametricLoss())(
        network_parameters(W, b), MODEL, (X, Y, μS))
    @test value isa Float32
    @test isapprox(value, closed_form_loss(W, b, X, Y, μS); rtol = 10 * eps(Float32))
    gradient = back(one(value))
    @test eltype(gradient.L1.W) === Float32
    @test eltype(gradient.L1.b) === Float32
end
