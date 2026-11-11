using GeometricMachineLearning
using GeometricMachineLearning: ForwardDiff, params
using GeometricOptimizers: Static, optimization_step!
using LinearAlgebra
using NeuralNetworkParameters: flatten, unflatten
using Test
import Random
import Zygote

include("../helpers/eltypes.jl")
include("../helpers/poisson.jl")

Random.seed!(1234)

# Φ and Ψ of a parameter tuple, formed densely, against which the layers' structured products are checked.
dense_phi(ps) = Matrix(ps.weight) * ps.factor
dense_psi(ps) = Matrix(ps.weight) * inv(ps.factor)' + (I - Matrix(ps.weight) * Matrix(ps.weight)') * ps.complement

# Move the parameters away from the start R = I, W = 0, at which the layer is a PSDLayer.
function perturb!(ps, T)
    ps.factor .+= T(0.3) * randn(T, size(ps.factor))
    ps.complement .+= randn(T, size(ps.complement))
    ps
end

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
    ps = params(NeuralNetwork(Chain(l), CPU(), T)).L1
    @test keys(ps) == (:weight, :factor, :complement)
    @test ps.factor == I && iszero(ps.complement)
    @test eltype(ps.factor) === T && eltype(ps.complement) === T
    ξ = randn(T, 4, 3)
    @test l(ξ, ps) ≈ PSDLayer(4, 10)(ξ, (weight = ps.weight,))
end

@testset "BiorthogonalPSDLayer: ΨᵀΦ = I, left inverse, symplectic ($T)" for T in REAL_ELTYPES
    l = BiorthogonalPSDLayer(4, 10)
    l_inv = inverse_layer(l)
    ps = perturb!(params(NeuralNetwork(Chain(l), CPU(), T)).L1, T)
    Φ, Ψ = dense_phi(ps), dense_psi(ps)
    @test norm(Ψ' * Φ - I) <= 100 * eps(T) * cond(ps.factor)
    # unlike the weight of a PSDLayer, Φ is not orthonormal
    @test norm(Φ' * Φ - I) > sqrt(eps(T))

    for ξ in (randn(T, 4), randn(T, 4, 3), randn(T, 4, 3, 2))
        z = @inferred l(ξ, ps)
        @test eltype(z) === T
        ξ_rec = @inferred l_inv(z, ps)
        @test norm(ξ_rec - ξ) <= 100 * eps(T) * cond(ps.factor) * max(one(T), norm(ξ))
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

    A = ForwardDiff.jacobian(x -> l(x, ps), ξ)
    J4, J10 = poisson_matrix(4, T), poisson_matrix(10, T)
    @test norm(A' * J10 * A - J4) <= 100 * eps(T) * cond(ps.factor)
    B = ForwardDiff.jacobian(x -> l_inv(x, ps), z)
    # the symplectic inverse A⁺ = J₄ᵀAᵀJ₁₀
    @test B ≈ J4' * A' * J10
end

@testset "BiorthogonalPSDLayer Zygote gradient" begin
    l = BiorthogonalPSDLayer(4, 10)
    ps0 = perturb!(params(NeuralNetwork(Chain(l), CPU(), Float64)).L1, Float64)
    ξ, z = randn(4, 5), randn(10, 5)
    loss(ps) = sum(abs2, l(ξ, ps)) + sum(abs2, inverse_layer(l)(z, ps))
    g = Zygote.gradient(loss, ps0)[1]
    for key in (:factor, :complement)
        fd = similar(getproperty(ps0, key))
        for i in eachindex(fd)
            x = getproperty(ps0, key)
            x0 = x[i]
            x[i] = x0 + 1e-6; fp = loss(ps0)
            x[i] = x0 - 1e-6; fm = loss(ps0)
            x[i] = x0
            fd[i] = (fp - fm) / 2e-6
        end
        @test isapprox(getproperty(g, key), fd; rtol = 1e-6)
    end
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

    for k in keys(params(nn))
        haskey(params(nn)[k], :complement) && perturb!(params(nn)[k], T)
    end
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
    # still a left inverse after training
    ξm = randn(T, 4, 5)
    @test norm(encoder(nn)(decoder(nn)(ξm)) - ξm) <= 1000 * eps(T) * norm(ξm)
end
