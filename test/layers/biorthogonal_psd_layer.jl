using GeometricMachineLearning
using GeometricMachineLearning: ForwardDiff, params
using AbstractNeuralNetworks: initialparameters, GlorotUniform
using GeometricOptimizers: BiorthogonalManifold, Static, check, optimization_step!, rgrad
using LinearAlgebra
using Test
import Random
import Zygote

include("../helpers/eltypes.jl")
include("../helpers/poisson.jl")

Random.seed!(1234)

# Φ and Ψ of a parameter tuple, the two halves of the stored matrix [Φ Ψ], against which the layers'
# products are checked.
function dense_pair(ps)
    A = parent(ps.weight)
    n = size(A, 2) ÷ 2
    A[:, 1:n], A[:, (n + 1):(2n)]
end

# Move the weight of a parameter tuple from the start (U, U), at which the layer is a PSDLayer, to a
# point of the manifold that `rand` draws.
function move!(ps, T)
    copyto!(ps.weight, rand(BiorthogonalManifold{T}, size(ps.weight)...))
    ps
end

# The rounding of the product of two matrices of norms `‖Φ‖` and `‖Ψ‖`, relative to the input.
product_tol(Φ, Ψ, T) = 100 * eps(T) * opnorm(Φ) * opnorm(Ψ)

@testset "BiorthogonalPSDLayer construction" begin
    l = BiorthogonalPSDLayer(4, 10)
    @test l === BiorthogonalPSDLayer{4, 10}()
    @test inverse_layer(l) === TransposeBiorthogonalPSDLayer(10, 4)
    @test inverse_layer(inverse_layer(l)) === l
    @test inverse_layer(BiorthogonalPSDLayer{4, 10}) === TransposeBiorthogonalPSDLayer{10, 4}
    @test inverse_layer(TransposeBiorthogonalPSDLayer{10, 4}) === BiorthogonalPSDLayer{4, 10}
    @test inverse_layer(BiorthogonalPSDLayer) === TransposeBiorthogonalPSDLayer

    r1 = r"^(?:ArgumentError: )?BiorthogonalPSDLayer dimensions \(3, 4\) must be positive even integers\.$"
    r2 = r"^(?:ArgumentError: )?BiorthogonalPSDLayer dimensions \(10, 4\): the latent dimension 10 must not exceed the ambient dimension 4\.$"
    r3 = r"^(?:ArgumentError: )?TransposeBiorthogonalPSDLayer dimensions \(4, 10\): the latent dimension 10 must not exceed the ambient dimension 4\.$"
    r4 = r"^(?:ArgumentError: )?TransposeBiorthogonalPSDLayer dimensions \(:?a, 4\) must be integers\.$"
    @test_throws r1 BiorthogonalPSDLayer(3, 4)
    @test_throws r2 BiorthogonalPSDLayer(10, 4)
    @test_throws r3 TransposeBiorthogonalPSDLayer(4, 10)
    @test_throws r4 TransposeBiorthogonalPSDLayer{:a, 4}()

    # the dimension 2Nn - n² of the manifold of biorthogonal pairs: n² conditions ΨᵀΦ = I on 2Nn entries
    for (M, N) in ((2, 2), (2, 4), (4, 10), (6, 20))
        n, N2 = M ÷ 2, N ÷ 2
        @test parameterlength(BiorthogonalPSDLayer(M, N)) == 2 * N2 * n - n^2
        @test parameterlength(TransposeBiorthogonalPSDLayer(N, M)) == 2 * N2 * n - n^2
    end
end

@testset "BiorthogonalPSDLayer starts as the PSDLayer of its Stiefel weight ($T)" for T in REAL_ELTYPES
    l = BiorthogonalPSDLayer(4, 10)
    ps = initialparameters(Random.Xoshiro(1), GlorotUniform(), l, CPU(), T)
    @test keys(ps) == (:weight,)
    @test ps.weight isa BiorthogonalManifold{T}
    @test size(ps.weight) == (5, 4)
    # the point (U, U), with U the weight that PSDLayer draws from the same generator
    U = initialparameters(Random.Xoshiro(1), GlorotUniform(), PSDLayer(4, 10), CPU(), T).weight
    Φ, Ψ = dense_pair(ps)
    @test Φ == Ψ == U
    @test initialparameters(Random.Xoshiro(1), GlorotUniform(), inverse_layer(l), CPU(), T).weight ==
          ps.weight
    ξ = randn(T, 4, 3)
    @test l(ξ, ps) ≈ PSDLayer(4, 10)(ξ, (weight = U,))
    @test inverse_layer(l)(l(ξ, ps), ps) ≈ ξ
end

@testset "BiorthogonalPSDLayer: ΨᵀΦ = I, left inverse, symplectic ($T)" for T in REAL_ELTYPES
    l = BiorthogonalPSDLayer(4, 10)
    l_inv = inverse_layer(l)
    ps = move!(params(NeuralNetwork(Chain(l), CPU(), T)).L1, T)
    @test ps.weight isa BiorthogonalManifold{T}
    Φ, Ψ = dense_pair(ps)
    # `rand` forms Ψ through a Cholesky factor of ΦᵀΦ, whose condition number is cond(Φ)²
    @test check(ps.weight) <= 100 * eps(T) * cond(Φ)^2
    # unlike the weight of a PSDLayer, Φ is not orthonormal
    @test norm(Φ' * Φ - I) > sqrt(eps(T))

    tol = check(ps.weight) + product_tol(Φ, Ψ, T)
    for ξ in (randn(T, 4), randn(T, 4, 3), randn(T, 4, 3, 2))
        z = @inferred l(ξ, ps)
        @test eltype(z) === T
        ξ_rec = @inferred l_inv(z, ps)
        @test norm(ξ_rec - ξ) <= tol * norm(ξ)
    end
    # against the dense matrices
    ξ = randn(T, 4)
    @test l(ξ, ps) ≈ vcat(Φ * ξ[1:2], Ψ * ξ[3:4])
    z = randn(T, 10)
    @test l_inv(z, ps) ≈ vcat(Ψ' * z[1:5], Φ' * z[6:10])
    # (q, p) input
    ξnt = (q = ξ[1:2], p = ξ[3:4])
    znt = @inferred l(ξnt, ps)
    @test vcat(znt.q, znt.p) ≈ l(ξ, ps)
    ξnt_rec = @inferred l_inv(znt, ps)
    @test norm(vcat(ξnt_rec.q, ξnt_rec.p) - ξ) <= tol * norm(ξ)

    A = ForwardDiff.jacobian(x -> l(x, ps), ξ)
    J4, J10 = poisson_matrix(4, T), poisson_matrix(10, T)
    # AᵀJA - J = diag(ΨᵀΦ - I, ...) blocks, plus the rounding of the products
    @test norm(A' * J10 * A - J4) <= 2 * tol
    B = ForwardDiff.jacobian(x -> l_inv(x, ps), z)
    # the symplectic inverse A⁺ = J₄ᵀAᵀJ₁₀
    @test B ≈ J4' * A' * J10
end

@testset "BiorthogonalPSDLayer Zygote gradient of the weight" begin
    l = BiorthogonalPSDLayer(4, 10)
    ps0 = move!(params(NeuralNetwork(Chain(l), CPU(), Float64)).L1, Float64)
    ξ, z = randn(4, 5), randn(10, 5)
    loss(ps) = sum(abs2, l(ξ, ps)) + sum(abs2, inverse_layer(l)(z, ps))
    g = Zygote.gradient(loss, ps0)[1]
    # the derivative with respect to the stored matrix [Φ Ψ], the free parameters of the point
    @test g.weight isa Matrix{Float64}
    @test size(g.weight) == size(ps0.weight)
    A = parent(ps0.weight)
    fd = similar(A)
    for i in eachindex(fd)
        a0 = A[i]
        A[i] = a0 + 1e-6; fp = loss(ps0)
        A[i] = a0 - 1e-6; fm = loss(ps0)
        A[i] = a0
        fd[i] = (fp - fm) / 2e-6
    end
    @test isapprox(g.weight, fd; rtol = 1e-6)
    # the Riemannian gradient is tangent: ΨᵀΔΦ + ΔΨᵀΦ = 0
    Φ, Ψ = dense_pair(ps0)
    Δ = rgrad(ps0.weight, g.weight)
    @test norm(Ψ' * Δ[:, 1:2] + Δ[:, 3:4]' * Φ) <= 1e-12 * norm(Δ) * opnorm(Φ) * opnorm(Ψ)
end

@testset "LeftInverseSymplecticAutoencoder(...; biorthogonal = true) ($T)" for T in REAL_ELTYPES
    arch = LeftInverseSymplecticAutoencoder(
        10, 4; n_blocks = 2, n_layers = 2, n_ambient_layers = 2, sympnet_upscale = 3,
        biorthogonal = true)
    @test arch isa LeftInverseSymplecticAutoencoder{:InitQ, typeof(tanh), :Biorthogonal}
    @test LeftInverseSymplecticAutoencoder(10, 4) isa
          LeftInverseSymplecticAutoencoder{:InitQ, typeof(tanh), :Stiefel}
    nn = NeuralNetwork(arch, CPU(), T)
    dec, enc = decoder(nn), encoder(nn)
    @test count(l -> l isa BiorthogonalPSDLayer, dec.model.layers) == 1
    @test !any(l -> l isa PSDLayer, dec.model.layers)
    @test enc.model.layers == reverse(map(inverse_layer, dec.model.layers))
    # the decoder of the Stiefel architecture plus what the biorthogonal lift adds
    n_stiefel = parameterlength(NeuralNetwork(
        LeftInverseSymplecticAutoencoder(
            10, 4; n_blocks = 2, n_layers = 2, n_ambient_layers = 2, sympnet_upscale = 3),
        CPU(), T))
    @test parameterlength(nn) ==
          n_stiefel - parameterlength(PSDLayer(4, 10)) + parameterlength(BiorthogonalPSDLayer(4, 10))

    lift_keys = [k for k in keys(params(nn)) if haskey(params(nn)[k], :weight) &&
                                                 params(nn)[k].weight isa BiorthogonalManifold]
    @test length(lift_keys) == 1
    lift = params(nn)[only(lift_keys)]
    move!(lift, T)
    for ξ in (randn(T, 4), randn(T, 4, 5), randn(T, 4, 5, 2))
        rec = enc(dec(ξ))
        @test eltype(rec) === T
        @test norm(rec - ξ) <= 1000 * eps(T) * max(one(T), norm(ξ))
    end
    ξv = randn(T, 4)
    D = ForwardDiff.jacobian(x -> dec(x), ξv)
    @test norm(D' * poisson_matrix(10, T) * D - poisson_matrix(4, T)) <= 1000 * eps(T)

    batch = randn(T, 10, 32)
    loss = AutoEncoderLoss()
    loss_init = loss(nn, batch)
    opt = TrainingOptimizer(params(nn); algorithm = Adam(), linesearch = Static(T(1e-2)))
    for _ in 1:5
        gs = Zygote.gradient(p -> loss(nn.model, p, batch), params(nn))[1]
        optimization_step!(params(nn), opt, gs)
    end
    @test loss(nn, batch) < loss_init
    # still a point of the manifold, and still a left inverse, after training
    Φ, Ψ = dense_pair(lift)
    @test check(lift.weight) <= 1000 * eps(T) * opnorm(Φ) * opnorm(Ψ)
    ξm = randn(T, 4, 5)
    @test norm(encoder(nn)(decoder(nn)(ξm)) - ξm) <= 1000 * eps(T) * norm(ξm)
end
