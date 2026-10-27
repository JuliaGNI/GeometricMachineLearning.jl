using FiniteDifferences
using GeometricMachineLearning
using GeometricMachineLearning: ForwardDiff, SymplecticCompression, params
using KernelAbstractions
using LinearAlgebra
using NeuralNetworkParameters: flatten, unflatten
using Test
import Random
import Zygote

include("../helpers/eltypes.jl")
include("../helpers/poisson.jl")

Random.seed!(1234)

@testset "WeightTiedSymplecticAutoencoder validation and structure" begin
    @test WeightTiedSymplecticAutoencoder <: SymplecticCompression

    re1 = r"^full_dim \(9\) and reduced_dim \(4\) must be positive even integers\.$"
    re2 = r"^full_dim \(10\) and reduced_dim \(3\) must be positive even integers\.$"
    re3 = r"^full_dim \(0\) and reduced_dim \(4\) must be positive even integers\.$"
    re4 = r"^full_dim \(10\) and reduced_dim \(0\) must be positive even integers\.$"
    re5 = r"^full_dim \(4\) must be greater than or equal to reduced_dim \(6\)\.$"
    re6 = r"^n_blocks \(0\) must be at least 1\.$"
    s7a = "^n_layers \\(-1\\) and n_ambient_layers \\(2\\) must be non-negative "
    s7b = "and sympnet_upscale \\(5\\) must be at least 1\\.\$"
    re7 = Regex(s7a * s7b)
    @test_throws re1 WeightTiedSymplecticAutoencoder(9, 4)
    @test_throws re2 WeightTiedSymplecticAutoencoder(10, 3)
    @test_throws re3 WeightTiedSymplecticAutoencoder(0, 4)
    @test_throws re4 WeightTiedSymplecticAutoencoder(10, 0)
    @test_throws re5 WeightTiedSymplecticAutoencoder(4, 6)
    @test_throws re6 WeightTiedSymplecticAutoencoder(10, 4; n_blocks = 0)
    @test_throws re7 WeightTiedSymplecticAutoencoder(10, 4; n_layers = -1)

    # Single-block case (n_blocks == 1) and equal-dimension case (full_dim == reduced_dim)
    arch1 = WeightTiedSymplecticAutoencoder(10, 4; n_blocks = 1, init_q = false)
    nn1 = NeuralNetwork(arch1, CPU(), Float64)
    dec1 = decoder(nn1)
    enc1 = encoder(nn1)
    @test length(dec1.model.layers) == 3 + 1 + 2
    @test dec1.model.layers[1] isa GradientLayerP{4, 4}
    @test dec1.model.layers[4] === PSDLayer(4, 10)
    @test dec1.model.layers[5] isa GradientLayerP{10, 10}
    @test enc1.model.layers == reverse(map(inverse_layer, dec1.model.layers))

    arch_eq = WeightTiedSymplecticAutoencoder(6, 6; n_blocks = 2, n_layers = 0)
    @test length(decoder(NeuralNetwork(arch_eq, CPU(), Float64)).model.layers) == 4

    # Multi-block weight tying and leaf identity (catches M2 and M4)
    arch = WeightTiedSymplecticAutoencoder(10, 4; n_blocks = 3)
    nn = NeuralNetwork(arch, CPU(), Float64)
    dec = decoder(nn)
    enc = encoder(nn)

    @test parameterlength(nn) == parameterlength(dec)
    @test enc.model.layers == reverse(map(inverse_layer, dec.model.layers))
    K = length(dec.params)
    @test length(enc.params) == K
    for k in 1:K
        @test enc.params[k] === dec.params[K - k + 1]
        @test dec.params[k] === nn.params[k]
        for key in keys(enc.params[k])
            @test enc.params[k][key] === dec.params[K - k + 1][key]
        end
    end
end

@testset "Round-trip and symplecticity ($T)" for T in REAL_ELTYPES
    for nb in (1, 2, 3)
        for iq in (true, false)
            arch = WeightTiedSymplecticAutoencoder(10, 4; n_blocks = nb, init_q = iq)
            nn = NeuralNetwork(arch, CPU(), T)
            enc = encoder(nn)
            dec = decoder(nn)

            ξv = randn(T, 4)
            ξm = randn(T, 4, 5)
            ξnt_v = (q = ξv[1:2], p = ξv[3:4])
            ξnt_m = (q = ξm[1:2, :], p = ξm[3:4, :])

            @test norm(enc(dec(ξv)) - ξv) <= 50 * eps(T) * norm(ξv)
            @test norm(enc(dec(ξm)) - ξm) <= 50 * eps(T) * norm(ξm)

            rec_v = enc(dec(ξnt_v))
            err_v = hypot(norm(rec_v.q - ξnt_v.q), norm(rec_v.p - ξnt_v.p))
            nrm_v = hypot(norm(ξnt_v.q), norm(ξnt_v.p))
            @test err_v <= 50 * eps(T) * nrm_v

            rec_m = enc(dec(ξnt_m))
            err_m = hypot(norm(rec_m.q - ξnt_m.q), norm(rec_m.p - ξnt_m.p))
            nrm_m = hypot(norm(ξnt_m.q), norm(ξnt_m.p))
            @test err_m <= 50 * eps(T) * nrm_m

            zv = dec(ξv)
            zm = dec(ξm)
            @test norm(nn(zv) - zv) <= 50 * eps(T) * norm(zv)
            @test norm(nn(zm) - zm) <= 50 * eps(T) * norm(zm)
        end
    end

    arch = WeightTiedSymplecticAutoencoder(10, 4)
    nn = NeuralNetwork(arch, CPU(), T)
    dec = decoder(nn)
    ξ = randn(T, 4)
    D = ForwardDiff.jacobian(x -> dec(x), ξ)
    J10 = poisson_matrix(T, 5)
    J4 = poisson_matrix(T, 2)
    @test norm(transpose(D) * J10 * D - J4) <= 100 * eps(T)
end

@testset "WeightTiedSymplecticAutoencoder Zygote gradient and @inferred" begin
    arch = WeightTiedSymplecticAutoencoder(6, 4; n_layers = 2, n_ambient_layers = 1)
    nn = NeuralNetwork(arch, CPU(), Float64)
    z = randn(Float64, 6, 4)
    loss = AutoEncoderLoss()

    gs_zyg = Zygote.gradient(ps -> loss(nn.model, ps, z), nn.params)[1]
    @test gs_zyg isa NetworkParameters

    v, layout = flatten(nn.params)
    v_zyg, _ = flatten(gs_zyg)
    fdm = central_fdm(5, 1)
    v_fd = grad(fdm, w -> loss(nn.model, unflatten(layout, w), z), v)[1]
    @test isapprox(v_zyg, v_fd; rtol = 10 * sqrt(eps(Float64)))

    for T in REAL_ELTYPES
        nn_T = NeuralNetwork(arch, CPU(), T)
        enc_T = encoder(nn_T)
        dec_T = decoder(nn_T)
        zv = randn(T, 6)
        ξv = randn(T, 4)
        znt = (q = zv[1:3], p = zv[4:6])
        ξnt = (q = ξv[1:2], p = ξv[3:4])

        y_enc = @inferred enc_T(zv)
        y_dec = @inferred dec_T(ξv)
        y_nn = @inferred nn_T(zv)
        @test y_enc isa Vector{T}
        @test y_dec isa Vector{T}
        @test y_nn isa Vector{T}

        nt_enc = @inferred enc_T(znt)
        nt_dec = @inferred dec_T(ξnt)
        nt_nn = @inferred nn_T(znt)
        @test nt_enc isa NamedTuple{(:q, :p), Tuple{Vector{T}, Vector{T}}}
        @test nt_dec isa NamedTuple{(:q, :p), Tuple{Vector{T}, Vector{T}}}
        @test nt_nn isa NamedTuple{(:q, :p), Tuple{Vector{T}, Vector{T}}}
    end
end
