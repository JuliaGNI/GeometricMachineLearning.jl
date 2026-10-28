using GeometricMachineLearning
using GeometricMachineLearning: ForwardDiff, SymplecticCompression,
                                compute_decoder_iterations,
                                decoder_layers_from_iteration, params
using GeometricOptimizers: Static, optimization_step!
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

    re1 = r"^(?:ArgumentError: )?full_dim \(9\) and reduced_dim \(4\) must be positive even integers\.$"
    re2 = r"^(?:ArgumentError: )?full_dim \(10\) and reduced_dim \(3\) must be positive even integers\.$"
    re3 = r"^(?:ArgumentError: )?full_dim \(0\) and reduced_dim \(4\) must be positive even integers\.$"
    re4 = r"^(?:ArgumentError: )?full_dim \(10\) and reduced_dim \(0\) must be positive even integers\.$"
    re5 = r"^(?:ArgumentError: )?full_dim \(4\) must be greater than or equal to reduced_dim \(6\)\.$"
    re6 = r"^(?:ArgumentError: )?n_blocks \(0\) must be at least 1\.$"
    s7a = "^(?:ArgumentError: )?n_layers \\(-1\\) and n_ambient_layers \\(2\\) must be non-negative "
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
    arch1 = WeightTiedSymplecticAutoencoder(
        4, 2; n_blocks = 1, n_layers = 1, n_ambient_layers = 2,
        sympnet_upscale = 1, init_q = true)
    nn1 = NeuralNetwork(arch1, CPU(), Float64)
    dec1 = decoder(nn1)
    enc1 = encoder(nn1)
    @test length(dec1.model.layers) == 1 + 1 + 2
    @test dec1.model.layers[1] isa GradientLayerQ{2, 2}
    @test dec1.model.layers[2] === PSDLayer(2, 4)
    @test dec1.model.layers[3] isa GradientLayerQ{4, 4}
    @test dec1.model.layers[4] isa GradientLayerP{4, 4}
    @test enc1.model.layers == reverse(map(inverse_layer, dec1.model.layers))

    arch_eq = WeightTiedSymplecticAutoencoder(6, 6; n_blocks = 2, n_layers = 0)
    @test compute_decoder_iterations(arch_eq) == [6, 6, 6]
    @test length(decoder_layers_from_iteration(arch_eq, compute_decoder_iterations(arch_eq))) ==
          4
    @test compute_decoder_iterations(WeightTiedSymplecticAutoencoder(10, 4; n_blocks = 3)) ==
          [4, 6, 8, 10]

    # Multi-block weight tying and leaf identity
    arch2 = WeightTiedSymplecticAutoencoder(
        6, 2; n_blocks = 2, n_layers = 2, n_ambient_layers = 0,
        sympnet_upscale = 1, init_q = false)
    nn = NeuralNetwork(arch2, CPU(), Float64)
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
    arch1 = WeightTiedSymplecticAutoencoder(
        4, 2; n_blocks = 1, n_layers = 1, n_ambient_layers = 2,
        sympnet_upscale = 1, init_q = true)
    nn = NeuralNetwork(arch1, CPU(), T)
    enc = encoder(nn)
    dec = decoder(nn)

    ξv = randn(T, 2)
    ξm = randn(T, 2, 5)
    ξnt_v = (q = ξv[1:1], p = ξv[2:2])
    ξnt_m = (q = ξm[1:1, :], p = ξm[2:2, :])

    rec_vec = enc(dec(ξv))
    rec_mat = enc(dec(ξm))
    @test eltype(rec_vec) === T
    @test eltype(rec_mat) === T
    @test norm(rec_vec - ξv) <= 50 * eps(T) * norm(ξv)
    @test norm(rec_mat - ξm) <= 50 * eps(T) * norm(ξm)

    rec_v = enc(dec(ξnt_v))
    @test eltype(rec_v.q) === T && eltype(rec_v.p) === T
    err_v = hypot(norm(rec_v.q - ξnt_v.q), norm(rec_v.p - ξnt_v.p))
    nrm_v = hypot(norm(ξnt_v.q), norm(ξnt_v.p))
    @test err_v <= 50 * eps(T) * nrm_v

    rec_m = enc(dec(ξnt_m))
    @test eltype(rec_m.q) === T && eltype(rec_m.p) === T
    err_m = hypot(norm(rec_m.q - ξnt_m.q), norm(rec_m.p - ξnt_m.p))
    nrm_m = hypot(norm(ξnt_m.q), norm(ξnt_m.p))
    @test err_m <= 50 * eps(T) * nrm_m

    zv = dec(ξv)
    zm = dec(ξm)
    @test eltype(zv) === T && eltype(zm) === T
    @test eltype(nn(zv)) === T && eltype(nn(zm)) === T
    @test norm(nn(zv) - zv) <= 50 * eps(T) * norm(zv)
    @test norm(nn(zm) - zm) <= 50 * eps(T) * norm(zm)

    D = ForwardDiff.jacobian(x -> dec(x), ξv)
    J4 = poisson_matrix(4, T)
    J2 = poisson_matrix(2, T)
    @test norm(transpose(D) * J4 * D - J2) <= 100 * eps(T)

    arch2 = WeightTiedSymplecticAutoencoder(
        6, 2; n_blocks = 2, n_layers = 2, n_ambient_layers = 0,
        sympnet_upscale = 1, init_q = false)
    nn2 = NeuralNetwork(arch2, CPU(), T)
    enc2 = encoder(nn2)
    dec2 = decoder(nn2)
    @test norm(enc2(dec2(ξv)) - ξv) <= 50 * eps(T) * norm(ξv)
    z2 = dec2(ξv)
    @test norm(nn2(z2) - z2) <= 50 * eps(T) * norm(z2)

    # (2N, 2n) = (10, 4) round-trip, Jacobian symplecticity, and 5-step training
    arch10 = WeightTiedSymplecticAutoencoder(
        10, 4; n_blocks = 2, n_layers = 2, n_ambient_layers = 2, sympnet_upscale = 3)
    nn10 = NeuralNetwork(arch10, CPU(), T)
    enc10 = encoder(nn10)
    dec10 = decoder(nn10)
    for shape in ((4,), (4, 5), (4, 5, 2))
        ξ10 = randn(T, shape...)
        rec10 = enc10(dec10(ξ10))
        @test eltype(rec10) === T
        @test norm(rec10 - ξ10) <= 100 * eps(T) * max(one(T), norm(ξ10))
    end
    ξ10_v = randn(T, 4)
    D10 = ForwardDiff.jacobian(x -> dec10(x), ξ10_v)
    J10 = poisson_matrix(10, T)
    @test norm(transpose(D10) * J10 * D10 - J4) <= 100 * eps(T)

    t_grid = range(zero(T), T(2π); length = 32)
    q_ho = zeros(T, 5, 32)
    p_ho = zeros(T, 5, 32)
    for k in 1:32
        q_ho[1, k] = cos(t_grid[k])
        p_ho[1, k] = -sin(t_grid[k])
        q_ho[2, k] = T(0.5) * sin(2 * t_grid[k])
        p_ho[2, k] = cos(2 * t_grid[k])
    end
    batch = vcat(q_ho, p_ho)
    loss = AutoEncoderLoss()
    loss_init = loss(nn10, batch)
    opt = TrainingOptimizer(params(nn10); algorithm = Adam(), linesearch = Static(T(1e-2)))
    for _ in 1:5
        gs = Zygote.gradient(p -> loss(nn10.model, p, batch), params(nn10))[1]
        optimization_step!(params(nn10), opt, gs)
    end
    loss_final = loss(nn10, batch)
    @test eltype(loss_init) === T && eltype(loss_final) === T
    @test loss_final < loss_init
end

@testset "WeightTiedSymplecticAutoencoder Zygote gradient and @inferred" begin
    arch = WeightTiedSymplecticAutoencoder(
        4, 2; n_blocks = 1, n_layers = 1, n_ambient_layers = 1,
        sympnet_upscale = 1, init_q = true)
    nn = NeuralNetwork(arch, CPU(), Float64)
    z = randn(Float64, 4, 3)
    loss = AutoEncoderLoss()

    gs_zyg = Zygote.gradient(ps -> loss(nn.model, ps, z), nn.params)[1]
    @test gs_zyg isa NetworkParameters

    v, layout = flatten(nn.params)
    v_zyg, _ = flatten(gs_zyg)
    eps_fd = cbrt(eps(Float64))
    v_fd = similar(v)
    vp = copy(v)
    vm = copy(v)
    for i in eachindex(v)
        vp[i] = v[i] + eps_fd
        vm[i] = v[i] - eps_fd
        f_plus = loss(nn.model, unflatten(layout, vp), z)
        f_minus = loss(nn.model, unflatten(layout, vm), z)
        v_fd[i] = (f_plus - f_minus) / (2 * eps_fd)
        vp[i] = v[i]
        vm[i] = v[i]
    end
    @test isapprox(v_zyg, v_fd; rtol = 10 * sqrt(eps(Float64)))

    # @inferred checks
    for T in REAL_ELTYPES
        nn = NeuralNetwork(arch, CPU(), T)
        enc = encoder(nn)
        dec = decoder(nn)
        nn_T = nn
        enc_T = enc
        dec_T = dec
        zv = randn(T, 4)
        zm = randn(T, 4, 3)
        ξv = randn(T, 2)
        ξm = randn(T, 2, 3)
        znt = (q = zv[1:2], p = zv[3:4])
        ξnt = (q = ξv[1:1], p = ξv[2:2])

        y_enc = @inferred enc(zv)
        y_dec = @inferred dec(ξv)
        y_nn = @inferred nn(zv)
        ym_enc = @inferred enc(zm)
        ym_dec = @inferred dec(ξm)
        ym_nn = @inferred nn(zm)
        @inferred enc_T(zm)
        @inferred dec_T(ξm)
        @inferred nn_T(zm)
        @test y_enc isa Vector{T}
        @test y_dec isa Vector{T}
        @test y_nn isa Vector{T}
        @test ym_enc isa Matrix{T}
        @test ym_dec isa Matrix{T}
        @test ym_nn isa Matrix{T}

        nt_enc = @inferred enc_T(znt)
        nt_dec = @inferred dec_T(ξnt)
        nt_nn = @inferred nn_T(znt)
        @test nt_enc isa NamedTuple{(:q, :p), Tuple{Vector{T}, Vector{T}}}
        @test nt_dec isa NamedTuple{(:q, :p), Tuple{Vector{T}, Vector{T}}}
        @test nt_nn isa NamedTuple{(:q, :p), Tuple{Vector{T}, Vector{T}}}
    end
end
