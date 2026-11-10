using GeometricMachineLearning
using GeometricMachineLearning: ForwardDiff, SymplecticCompression, params
using GeometricMachineLearning.AbstractNeuralNetworks: GlorotUniform
using GeometricOptimizers: Static, optimization_step!
using KernelAbstractions
using LinearAlgebra
using NeuralNetworkParameters: flatten, unflatten
using Test
import Random
import Zygote

include("../helpers/eltypes.jl")
include("../helpers/poisson.jl")

Random.seed!(385)

# The relabelling of `M` particles in `ℝ^d` by `σ` as a matrix on the flattened state `(q, p)`, in
# which particle `i` is the block `(i - 1) * d + 1:i * d` of `q` and of `p`.
function particle_perm_matrix(σ::AbstractVector{<:Integer}, d::Int, ::Type{T}) where {T}
    M = length(σ)
    P = zeros(T, M * d, M * d)
    for i in 1:M
        j = σ[i]
        P[((i - 1) * d + 1):(i * d), ((j - 1) * d + 1):(j * d)] .= Matrix{T}(I, d, d)
    end
    [P zeros(T, M * d, M * d); zeros(T, M * d, M * d) P]
end

function permute_particles(qp::NamedTuple{(:q, :p)}, σ::AbstractVector{<:Integer})
    ndims(qp.q) == 2 ? (q = qp.q[:, σ], p = qp.p[:, σ]) :
    (q = qp.q[:, σ, :], p = qp.p[:, σ, :])
end

function qp_isapprox(a::NamedTuple, b::NamedTuple, tol)
    hypot(norm(a.q - b.q), norm(a.p - b.p)) <= tol * max(1, hypot(norm(b.q), norm(b.p)))
end

act_biased(x) = tanh(x) + oftype(x, 0.35) * cos(x)

@testset "ParticlePSDLayer and TransposeParticlePSDLayer" begin
    re_args = r"^(?:ArgumentError: )?ParticlePSDLayer requires M >= 1, d_in >= 1 and d_out >= 1, got M = 0, d_in = 1, d_out = 2\.$"
    re_up = r"^(?:ArgumentError: )?ParticlePSDLayer requires d_in <= d_out, got d_in = 3, d_out = 2\.$"
    re_down = r"^(?:ArgumentError: )?TransposeParticlePSDLayer requires d_in >= d_out, got d_in = 2, d_out = 3\.$"
    @test_throws re_args ParticlePSDLayer(0, 1, 2)
    @test_throws re_up ParticlePSDLayer(4, 3, 2)
    @test_throws re_down TransposeParticlePSDLayer(4, 2, 3)

    l = ParticlePSDLayer(4, 2, 3)
    lt = TransposeParticlePSDLayer(4, 3, 2)
    @test l isa ParticlePSDLayer{16, 24}
    @test lt isa TransposeParticlePSDLayer{24, 16}
    @test inverse_layer(l) == lt
    @test inverse_layer(lt) == l
    @test inverse_layer(typeof(l)) === typeof(lt)
    @test inverse_layer(typeof(lt)) === typeof(l)
    @test inverse_layer(ParticlePSDLayer) === TransposeParticlePSDLayer
    @test inverse_layer(TransposeParticlePSDLayer) === ParticlePSDLayer
    # dim St(2, 3) = 2 * 3 - 2 * 3 / 2
    @test parameterlength(l) == 3
    @test parameterlength(lt) == 3
    @test parameterlength(ParticlePSDLayer(4, 3, 3)) == 3

    @testset "Precision $T" for T in REAL_ELTYPES
        M, d_in, d_out = 4, 2, 3
        ps = initialparameters(Random.default_rng(), GlorotUniform(), l, CPU(), T)
        ps_t = initialparameters(Random.default_rng(), GlorotUniform(), lt, CPU(), T)
        Φ = ps.weight
        @test Φ isa StiefelManifold
        @test eltype(Φ) === T
        @test size(Φ) == (d_out, d_in)
        @test size(ps_t.weight) == (d_out, d_in)
        @test norm(Φ' * Φ - I) <= 100 * eps(T)

        # Each particle is lifted with the same Φ.
        q = randn(T, d_in, M)
        p = randn(T, d_in, M)
        z = vcat(vec(q), vec(p))
        y = @inferred l(z, ps)
        @test eltype(y) === T
        @test isapprox(y, vcat(vec(Matrix(Φ) * q), vec(Matrix(Φ) * p));
            rtol = 100 * eps(T))

        # The flattened and the particle layout give the same result, also for a batch.
        y_qp = @inferred l((q = q, p = p), ps)
        @test size(y_qp.q) == (d_out, M)
        @test vcat(vec(y_qp.q), vec(y_qp.p)) ≈ y
        B = 3
        qb = randn(T, d_in, M, B)
        pb = randn(T, d_in, M, B)
        zb = vcat(reshape(qb, M * d_in, B), reshape(pb, M * d_in, B))
        yb = @inferred l(zb, ps)
        yb_qp = @inferred l((q = qb, p = pb), ps)
        @test size(yb) == (2 * M * d_out, B)
        @test size(yb_qp.q) == (d_out, M, B)
        @test yb ≈ vcat(reshape(yb_qp.q, M * d_out, B), reshape(yb_qp.p, M * d_out, B))

        # The transpose with the same parameters is a left inverse, and the lift is symplectic.
        @test norm(lt(y, ps) - z) <= 100 * eps(T) * norm(z)
        @test norm(lt(yb, ps) - zb) <= 100 * eps(T) * norm(zb)
        A = ForwardDiff.jacobian(x -> l(x, ps), z)
        @test norm(transpose(A) * poisson_matrix(2 * M * d_out, T) * A -
                   poisson_matrix(2 * M * d_in, T)) <= 100 * eps(T)

        # The lift commutes with every relabelling.
        σ = Random.randperm(M)
        @test l(particle_perm_matrix(σ, d_in, T) * z, ps) ≈ particle_perm_matrix(σ, d_out, T) * y

        @test_throws DimensionMismatch l(randn(T, 2 * M * d_in + 2), ps)
        @test_throws DimensionMismatch l((q = randn(T, 5), p = randn(T, 5)), ps)
        @test_throws DimensionMismatch l((q = randn(T, d_in, M), p = randn(T, d_in, M + 1)), ps)
    end
end

@testset "EquivariantSymplecticAutoencoder validation and structure" begin
    @test EquivariantSymplecticAutoencoder <: SymplecticCompression

    re1 = r"^(?:ArgumentError: )?EquivariantSymplecticAutoencoder requires M >= 1, d >= 1 and 1 <= d_latent <= d, got M = 4, d = 2, d_latent = 3\.$"
    re2 = r"^(?:ArgumentError: )?EquivariantSymplecticAutoencoder requires M >= 1, d >= 1 and 1 <= d_latent <= d, got M = 0, d = 2, d_latent = 1\.$"
    re3 = r"^(?:ArgumentError: )?n_blocks \(0\) must be at least 1\.$"
    re4 = r"^(?:ArgumentError: )?n_blocks \(4\) must be at most 3 for d \(3\) and d_latent \(1\)\.$"
    re5 = r"^(?:ArgumentError: )?n_layers \(-1\) and n_ambient_layers \(2\) must be non-negative and upscale \(3\) must be at least 1\.$"
    re6 = r"^(?:ArgumentError: )?An EquivariantSymplecticAutoencoder has no Chain of untied layers: build it with NeuralNetwork\(arch\) and take encoder\(nn\) and decoder\(nn\)\.$"
    @test_throws re1 EquivariantSymplecticAutoencoder(4, 2, 3)
    @test_throws re2 EquivariantSymplecticAutoencoder(0, 2, 1)
    @test_throws re3 EquivariantSymplecticAutoencoder(4, 2, 1; n_blocks = 0)
    @test_throws re4 EquivariantSymplecticAutoencoder(4, 3, 1; n_blocks = 4)
    @test_throws re5 EquivariantSymplecticAutoencoder(4, 2, 1; n_layers = -1)
    @test_throws re6 Chain(EquivariantSymplecticAutoencoder(4, 2, 1))

    arch = EquivariantSymplecticAutoencoder(4, 3, 1)
    @test arch.full_dim == 24
    @test arch.reduced_dim == 8
    @test (arch.n_blocks, arch.n_layers, arch.n_ambient_layers, arch.upscale) == (2, 3, 2, 3)
    @test arch.activation === tanh
    @test arch isa EquivariantSymplecticAutoencoder{:InitQ}
    @test EquivariantSymplecticAutoencoder(4, 3, 1; init_q = false) isa
          EquivariantSymplecticAutoencoder{:InitP}
    # d == d_latent takes any n_blocks
    @test EquivariantSymplecticAutoencoder(4, 2, 2; n_blocks = 3).n_blocks == 3

    # The decoder: shears at d_k, a lift from d_k to d_{k + 1}, ..., and the ambient shears.
    arch = EquivariantSymplecticAutoencoder(
        4, 3, 1; n_blocks = 3, n_layers = 2, n_ambient_layers = 1, upscale = 2)
    nn = NeuralNetwork(arch, CPU(), Float64)
    dec_layers = nn.model.decoder_layers
    @test length(dec_layers) == 2 + 1 + 2 + 1 + 1
    @test dec_layers[1] isa PairwiseSymplecticLayerQ && dec_layers[1].particle_dim == 1
    @test dec_layers[2] isa PairwiseSymplecticLayerP && dec_layers[2].particle_dim == 1
    @test dec_layers[3] == ParticlePSDLayer(4, 1, 2)
    @test dec_layers[4] isa PairwiseSymplecticLayerQ && dec_layers[4].particle_dim == 2
    @test dec_layers[6] == ParticlePSDLayer(4, 2, 3)
    @test dec_layers[7] isa PairwiseSymplecticLayerQ && dec_layers[7].particle_dim == 3
    @test nn.model.encoder_layers == reverse(map(inverse_layer, dec_layers))
    @test parameterlength(nn) == parameterlength(decoder(nn)) == parameterlength(encoder(nn))
end

@testset "Equivariance, round trip and symplecticity ($T, init_q = $init_q)" for T in REAL_ELTYPES,
    init_q in (true, false)

    M, d, d_latent = 4, 3, 1
    arch = EquivariantSymplecticAutoencoder(
        M, d, d_latent; n_blocks = 3, n_layers = 2, n_ambient_layers = 2, upscale = 2,
        activation = act_biased, init_q = init_q)
    nn = NeuralNetwork(arch, CPU(), T)
    enc = encoder(nn)
    dec = decoder(nn)
    N, n = M * d, M * d_latent
    B = 3
    tol = 200 * eps(T)

    qp_s = (q = randn(T, d, M), p = randn(T, d, M))
    qp_b = (q = randn(T, d, M, B), p = randn(T, d, M, B))
    ξ_s = (q = randn(T, d_latent, M), p = randn(T, d_latent, M))
    ξ_b = (q = randn(T, d_latent, M, B), p = randn(T, d_latent, M, B))
    z_s = vcat(vec(qp_s.q), vec(qp_s.p))
    z_b = vcat(reshape(qp_b.q, N, B), reshape(qp_b.p, N, B))
    ξ_vec = vcat(vec(ξ_s.q), vec(ξ_s.p))
    ξ_mat = vcat(reshape(ξ_b.q, n, B), reshape(ξ_b.p, n, B))

    # Layouts: a (q, p) input gives a (q, p) output in the same layout.
    @test (@inferred enc(qp_s)).q isa Matrix{T}
    @test size(enc(qp_s).q) == (d_latent, M)
    @test size((@inferred enc(qp_b)).q) == (d_latent, M, B)
    @test size((@inferred dec(ξ_s)).q) == (d, M)
    @test size((@inferred dec(ξ_b)).q) == (d, M, B)
    @test (@inferred enc(z_s)) isa Vector{T}
    @test (@inferred enc(z_b)) isa Matrix{T}
    @test (@inferred dec(ξ_vec)) isa Vector{T}
    @test (@inferred dec(ξ_mat)) isa Matrix{T}
    @test (@inferred nn(z_b)) isa Matrix{T}
    @test size((@inferred nn(qp_b)).q) == (d, M, B)
    @test (@inferred enc((q = vec(qp_s.q), p = vec(qp_s.p)))).q isa Vector{T}
    @test enc(z_s) ≈ vcat(vec(enc(qp_s).q), vec(enc(qp_s).p))
    @test dec(ξ_mat) ≈ vcat(reshape(dec(ξ_b).q, N, B), reshape(dec(ξ_b).p, N, B))

    for _ in 1:5
        σ = Random.randperm(M)
        P_amb = particle_perm_matrix(σ, d, T)
        P_lat = particle_perm_matrix(σ, d_latent, T)

        # (i) the encoder maps a relabelled state to the relabelled latent particles
        @test qp_isapprox(enc(permute_particles(qp_s, σ)), permute_particles(enc(qp_s), σ), tol)
        @test qp_isapprox(enc(permute_particles(qp_b, σ)), permute_particles(enc(qp_b), σ), tol)
        @test norm(enc(P_amb * z_b) - P_lat * enc(z_b)) <= tol * max(1, norm(enc(z_b)))

        # (ii) the decoder maps relabelled latent particles to the relabelled state
        @test qp_isapprox(dec(permute_particles(ξ_s, σ)), permute_particles(dec(ξ_s), σ), tol)
        @test qp_isapprox(dec(permute_particles(ξ_b, σ)), permute_particles(dec(ξ_b), σ), tol)
        @test norm(dec(P_lat * ξ_mat) - P_amb * dec(ξ_mat)) <= tol * max(1, norm(dec(ξ_mat)))
    end

    # (iii) the encoder is a left inverse of the decoder
    @test norm(enc(dec(ξ_vec)) - ξ_vec) <= tol * max(1, norm(ξ_vec))
    @test norm(enc(dec(ξ_mat)) - ξ_mat) <= tol * max(1, norm(ξ_mat))
    @test qp_isapprox(enc(dec(ξ_s)), ξ_s, tol)
    @test qp_isapprox(enc(dec(ξ_b)), ξ_b, tol)
    z_dec = dec(ξ_mat)
    @test norm(nn(z_dec) - z_dec) <= tol * max(1, norm(z_dec))

    # (iv) the decoder is a symplectic embedding, and the encoder a Poisson map on its image
    J_N = poisson_matrix(2 * N, T)
    J_n = poisson_matrix(2 * n, T)
    D = ForwardDiff.jacobian(x -> dec(x), ξ_vec)
    @test size(D) == (2 * N, 2 * n)
    @test norm(transpose(D) * J_N * D - J_n) <= tol * max(1, norm(D)^2)
    E = ForwardDiff.jacobian(x -> enc(x), dec(ξ_vec))
    @test norm(E * J_N * transpose(E) - J_n) <= tol * max(1, norm(E)^2)
end

@testset "EquivariantSymplecticAutoencoder gradient and training" begin
    arch = EquivariantSymplecticAutoencoder(
        3, 2, 1; n_blocks = 1, n_layers = 1, n_ambient_layers = 1, upscale = 1)
    nn = NeuralNetwork(arch, CPU(), Float64)
    z = randn(Float64, 12, 3)
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
        v_fd[i] = (loss(nn.model, unflatten(layout, vp), z) -
                   loss(nn.model, unflatten(layout, vm), z)) / (2 * eps_fd)
        vp[i] = v[i]
        vm[i] = v[i]
    end
    @test isapprox(v_zyg, v_fd; rtol = 10 * sqrt(eps(Float64)))

    # A few Adam steps lower the loss and keep the shared Φ on the Stiefel manifold.
    @testset "Training ($T)" for T in REAL_ELTYPES
        M, d = 4, 3
        arch = EquivariantSymplecticAutoencoder(M, d, 1; n_layers = 1, n_ambient_layers = 1)
        nn = NeuralNetwork(arch, CPU(), T)
        data = randn(T, 2 * M * d, 8)
        loss_init = loss(nn, data)
        opt = TrainingOptimizer(params(nn); algorithm = Adam(), linesearch = Static(T(1e-3)))
        for _ in 1:5
            gs = Zygote.gradient(p -> loss(nn.model, p, data), params(nn))[1]
            optimization_step!(params(nn), opt, gs)
        end
        loss_final = loss(nn, data)
        @test eltype(loss_final) === T
        @test loss_final < loss_init
        for (layer, ps) in zip(nn.model.decoder_layers, values(params(nn)))
            if layer isa ParticlePSDLayer
                @test norm(ps.weight' * ps.weight - I) <= 100 * eps(T)
            end
        end
    end
end
