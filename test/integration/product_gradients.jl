using GeometricMachineLearning
using GeometricMachineLearning: _mul, params
using FiniteDifferences: central_fdm, grad
using Zygote: gradient
using Test
import Random

Random.seed!(123)

# The finite-difference gradient of `loss(W)` with respect to the storage `W.S` of a structured
# weight, in `Float64` on the `Float64` copy of the data, which is what the gradient of a
# `NetworkParameters` holds in its structured leaf.
function storage_fd(loss, W)
    grad(central_fdm(5, 1), S -> loss(typeof(W).name.wrapper(S, W.n)), Float64.(W.S))[1]
end

# `Float64` at `1e-8`, above the `1e-12` the products reach; `Float32` at its own accuracy.
tolerance(T) = T === Float64 ? 1e-8 : sqrt(eps(Float32))

# The network's single structured leaf replaced by `W`, for the finite-difference loss.
with_leaf(ps, W) = NetworkParameters((L1 = NamedTuple{keys(ps.L1)}((W,)),))

@testset "a structured weight in _mul: the storage gradient equals finite differences" begin
    for T in (Float32, Float64)
        x = rand(T, 4, 5, 3)
        z = rand(T, 3, 4, 5)
        for TT in (SymmetricMatrix, SkewSymMatrix, StrictlyLowerTriangular, StrictlyUpperTriangular)
            ps = NetworkParameters((L1 = (W = rand(TT{T}, 4),),))
            g = gradient(p -> sum(abs2, sin.(_mul(p.L1.W, x))), ps)[1].L1.W
            fd = storage_fd(W -> sum(abs2, sin.(_mul(W, Float64.(x)))), ps.L1.W)
            @test g isa TT{T}
            @test isapprox(g.S, fd; rtol = tolerance(T))
        end
        ps = NetworkParameters((L1 = (W = rand(SymmetricMatrix{T}, 4),),))
        g = gradient(p -> sum(abs2, sin.(_mul(z, p.L1.W))), ps)[1].L1.W
        fd = storage_fd(W -> sum(abs2, sin.(_mul(Float64.(z), W))), ps.L1.W)
        @test g isa SymmetricMatrix{T}
        @test isapprox(g.S, fd; rtol = tolerance(T))
    end
end

@testset "LinearLayerQ: the same gradient on a matrix and on the same data as a tensor" begin
    for T in (Float32, Float64)
        nn = NeuralNetwork(Chain(LinearLayerQ(4)), CPU(), T)
        ps = params(nn)
        x = rand(T, 4, 7)
        g_matrix = gradient(p -> sum(abs2, nn.model(x, p)), ps)[1].L1.weight
        g_tensor = gradient(p -> sum(abs2, nn.model(reshape(x, 4, 1, 7), p)), ps)[1].L1.weight
        fd = storage_fd(W -> sum(abs2, nn.model(Float64.(x), with_leaf(ps, W))), ps.L1.weight)
        @test g_matrix isa SymmetricMatrix{T}
        @test g_tensor isa SymmetricMatrix{T}
        @test isapprox(g_tensor.S, g_matrix.S; rtol = tolerance(T))
        @test isapprox(g_tensor.S, fd; rtol = tolerance(T))
    end
end

@testset "the symplectic attention layers on a tensor: the storage gradient equals finite differences" begin
    for T in (Float32, Float64),
        (layer, x) in ((LinearSymplecticAttentionQ(4, 5), rand(T, 4, 5, 3)),
            (SymplecticAttentionQ(4; symmetric = true), rand(T, 4, 3, 2)))

        nn = NeuralNetwork(Chain(layer), CPU(), T)
        ps = params(nn)
        g = only(values(gradient(p -> sum(abs2, nn.model(x, p)), ps)[1].L1))
        W = only(values(ps.L1))
        fd = storage_fd(W -> sum(abs2, nn.model(Float64.(x), with_leaf(ps, W))), W)
        @test g isa SymmetricMatrix{T}
        @test isapprox(g.S, fd; rtol = tolerance(T))
    end
end

# A Stiefel weight multiplied as `Y'` gets a dense `Matrix` gradient leaf, on matrix input as on
# tensor input, and the two agree on the same data.
@testset "a Stiefel weight multiplied as its adjoint: the gradient leaf is a Matrix" begin
    for T in (Float32, Float64)
        nn = NeuralNetwork(Chain(StiefelLayer(6, 3)), CPU(), T)
        ps = params(nn)
        x = rand(T, 6, 5)
        g_matrix = gradient(p -> sum(abs2, nn.model(x, p)), ps)[1].L1.weight
        g_tensor = gradient(p -> sum(abs2, nn.model(reshape(x, 6, 1, 5), p)), ps)[1].L1.weight
        @test g_matrix isa Matrix{T}
        @test g_tensor isa Matrix{T}
        @test isapprox(g_matrix, g_tensor; rtol = sqrt(eps(T)))
        @test isapprox(g_matrix, 2 * x * (Matrix(ps.L1.weight)' * x)'; rtol = sqrt(eps(T)))

        mha = NeuralNetwork(Chain(MultiHeadAttention(4, 2; Stiefel = true)), CPU(), T)
        g = gradient(p -> sum(abs2, mha.model(rand(T, 4, 3, 5), p)), params(mha))[1].L1
        @test all(head -> head isa Matrix{T}, (
            values(g.PQ)..., values(g.PK)..., values(g.PV)...))
    end
end
