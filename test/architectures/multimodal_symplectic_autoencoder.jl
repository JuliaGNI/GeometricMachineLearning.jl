using GeometricMachineLearning
using GeometricMachineLearning:
                                SymplecticCompression, AutoEncoder, Architecture,
                                NetworkLoss, params
using GeometricMachineLearning.AbstractNeuralNetworks: GlorotUniform, Initializer,
                                                       initialparameters
using KernelAbstractions
using LinearAlgebra
using Test
using Random: Random
using Zygote: Zygote

include("../helpers/eltypes.jl")

Random.seed!(42)

# A Glorot-uniform initializer that counts its calls, to see which networks it reaches.
struct CountingInitializer <: Initializer
    calls::Base.RefValue{Int}
end
function (c::CountingInitializer)(rng, x)
    c.calls[] += 1
    GlorotUniform()(rng, x)
end

# Helper to flatten and unflatten nested NamedTuple/Array parameters for finite differences
function _flatten_params(x::AbstractArray{T}) where {T <: Real}
    raw = parent(x)
    vec_copy = copy(vec(raw))
    sz = size(raw)
    is_stiefel = x isa StiefelManifold
    rebuild = v -> begin
        A = reshape(v, sz)
        is_stiefel ? StiefelManifold(A) : A
    end
    return vec_copy, rebuild
end

function _flatten_params(nt::NamedTuple{names}) where {names}
    vecs = Vector{Float64}[]
    rebuilders = Any[]
    lengths = Int[]
    for k in names
        v, r = _flatten_params(getfield(nt, k))
        push!(vecs, Float64.(v))
        push!(rebuilders, r)
        push!(lengths, length(v))
    end
    flat = isempty(vecs) ? Float64[] : reduce(vcat, vecs)
    rebuild = w -> begin
        offset = 0
        vals = map(eachindex(names)) do i
            len = lengths[i]
            sub = w[(offset + 1):(offset + len)]
            offset += len
            rebuilders[i](sub)
        end
        NamedTuple{names}(Tuple(vals))
    end
    return flat, rebuild
end

function _flatten_params(ps::NetworkParameters)
    flat, rebuild_nt = _flatten_params(params(ps))
    return flat, w -> NetworkParameters(rebuild_nt(w))
end

function _extract_grad_vec(g::NamedTuple{names}) where {names}
    vecs = Float64[]
    for k in names
        v = getfield(g, k)
        if v isa NamedTuple
            append!(vecs, _extract_grad_vec(v))
        elseif v isa AbstractArray
            append!(vecs, vec(Float64.(Array(v))))
        end
    end
    return vecs
end

_extract_grad_vec(g::NetworkParameters) = _extract_grad_vec(params(g))

@testset "Clause 1: MultiModalSymplecticAutoencoder construction and edge cases" begin
    @test isdefined(GeometricMachineLearning, :MultiModalSymplecticAutoencoder)
    @test isdefined(GeometricMachineLearning, :encode)
    @test isdefined(GeometricMachineLearning, :decode)
    @test isdefined(GeometricMachineLearning, :translate)
    @test MultiModalSymplecticAutoencoder <: SymplecticCompression
    @test MultiModalSymplecticAutoencoder <: AutoEncoder
    @test MultiModalSymplecticAutoencoder <: Architecture

    core = WeightTiedSymplecticAutoencoder(
        8, 4; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)

    re_empty = r"^(?:ArgumentError: )?MultiModalSymplecticAutoencoder requires a non-empty frontends NamedTuple\.$"
    @test_throws re_empty MultiModalSymplecticAutoencoder(NamedTuple(), core)

    fe_bad = CanonicalSortLayer(3, 2) # output dim = 12 != 8
    re_dim = r"^(?:ArgumentError: )?Frontend :particles has output dimension 12, expected core\.full_dim = 8\.$"
    @test_throws re_dim MultiModalSymplecticAutoencoder((particles = fe_bad,), core)

    fe_bad_grid = BilinearStiefelLayer(6, 6, 3, 2) # output dim = 2 * 3 * 2 = 12 != 8
    re_dim_grid = r"^(?:ArgumentError: )?Frontend :grid has output dimension 12, expected core\.full_dim = 8\.$"
    @test_throws re_dim_grid MultiModalSymplecticAutoencoder((grid = fe_bad_grid,), core)

    fe_part = CanonicalSortLayer(2, 2) # output dim = 8
    fe_grid = BilinearStiefelLayer(4, 4, 2, 2) # output dim = 8
    fe_dlra = LowRankStiefelLayer(4, 4, 2, 2; r_out = 4) # output dim = 8
    arch = MultiModalSymplecticAutoencoder(
        (particles = fe_part, grid = fe_grid, dlra = fe_dlra), core)
    @test arch.full_dim == 8
    @test arch.reduced_dim == 4

    nn = NeuralNetwork(arch, CPU(), Float64)
    re_unknown = r"^(?:ArgumentError: )?Unknown modality :unknown\. Available modalities: \(:particles, :grid, :dlra\)\.$"
    @test_throws re_unknown encode(nn, :unknown, zeros(8))
    @test_throws re_unknown decode(nn, :unknown, zeros(4))
    @test_throws re_unknown translate(nn, :unknown, :grid, zeros(8))
    @test_throws re_unknown translate(nn, :grid, :unknown, zeros(4, 4))
end

@testset "Clause 1: encode, decode, translate and right-invertible round-trip ($T)" for T in REAL_ELTYPES
    rng = Random.MersenneTwister(101)
    nx, nv = 5, 4
    mx, mv = 2, 2
    N = mx * mv # 4
    full_dim = 2 * N # 8
    reduced_dim = 4

    # Upper-triangular Cholesky metric factors for the spline modality
    A_x = randn(rng, T, nx, nx)
    A_v = randn(rng, T, nv, nv)
    R_x = Matrix{T}(cholesky(Symmetric(transpose(A_x) * A_x + T(2) * I)).U)
    R_v = Matrix{T}(cholesky(Symmetric(transpose(A_v) * A_v + T(2) * I)).U)

    fe_sort = CanonicalSortLayer(2, 2)
    fe_slot_cm = CanonicalSlotLayer(4, 4, 1) # NOut = 2 * 1 * 4 = 8
    fe_slot = CanonicalSlotLayer(3, 2, 2) # NOut = 2 * 2 * 2 = 8
    fe_spline = BilinearStiefelLayer(nx, nv, mx, mv; metric_factors = (R_x, R_v))
    fe_grid = BilinearStiefelLayer(nx, nv, mx, mv)
    fe_dlra = LowRankStiefelLayer(nx, nv, mx, mv; r_out = mx * mv)

    core = WeightTiedSymplecticAutoencoder(
        full_dim, reduced_dim;
        n_blocks = 1,
        n_layers = 1,
        n_ambient_layers = 0,
        sympnet_upscale = 2
    )
    frontends = (
        particles = fe_sort,
        particles_slot_cm = fe_slot_cm,
        spline = fe_spline,
        grid = fe_grid,
        dlra = fe_dlra
    )
    arch = MultiModalSymplecticAutoencoder(frontends, core)
    nn = NeuralNetwork(arch, CPU(), T; rng = rng)

    # 1. Verify encode, decode and translate match explicit frontend + core composition
    q_p = randn(rng, T, 2, 2)
    p_p = randn(rng, T, 2, 2)
    x_part = (q = q_p, p = p_p)
    F_grid = randn(rng, T, nx, nv)
    F_spline = randn(rng, T, nx, nv)
    r_in = 3
    X_in = Matrix(qr(randn(rng, T, nx, r_in)).Q[:, 1:r_in])
    V_in = Matrix(qr(randn(rng, T, nv, r_in)).Q[:, 1:r_in])
    S_in = randn(rng, T, r_in, r_in)
    x_dlra = (X = X_in, S = S_in, V = V_in)

    ξ_part = @inferred encode(nn, :particles, x_part)
    ξ_grid = @inferred encode(nn, Val(:grid), F_grid)
    ξ_spline = @inferred encode(nn, Val(:spline), F_spline)
    ξ_dlra = @inferred encode(nn, :dlra, x_dlra)
    @test isapprox(ξ_grid, encode(nn, :grid, F_grid); atol = 8 * eps(T), rtol = 8 * eps(T))
    @test isapprox(
        ξ_spline, encode(nn, :spline, F_spline); atol = 8 * eps(T), rtol = 8 * eps(T))

    @test eltype(ξ_part) === T && length(ξ_part) == reduced_dim
    @test eltype(ξ_grid) === T && length(ξ_grid) == reduced_dim
    @test eltype(ξ_spline) === T && length(ξ_spline) == reduced_dim
    @test eltype(ξ_dlra) === T && length(ξ_dlra) == reduced_dim

    # Explicit frontend followed by core encoder matches encode(nn, modality, x)
    z_grid_fe = fe_grid(F_grid, params(nn).frontends.grid)
    ξ_grid_manual = encoder(nn)(vcat(z_grid_fe.q, z_grid_fe.p))
    @test isapprox(ξ_grid, ξ_grid_manual; atol = 8 * eps(T), rtol = 8 * eps(T))

    # Explicit core decoder followed by inverse_layer(frontend) matches decode(nn, modality, ξ)
    z_dec_manual = decoder(nn)(ξ_grid)
    F_grid_manual = inverse_layer(fe_grid)(
        (q = z_dec_manual[1:N], p = z_dec_manual[(N + 1):full_dim]),
        params(nn).frontends.grid
    )
    F_grid_dec = @inferred decode(nn, Val(:grid), ξ_grid)
    @test isapprox(
        F_grid_dec, decode(nn, :grid, ξ_grid); atol = 8 * eps(T), rtol = 8 * eps(T))
    @test isapprox(F_grid_dec, F_grid_manual; atol = 8 * eps(T), rtol = 8 * eps(T))

    # NamedTuple (q, p) latent input to decode
    F_grid_dec_nt = decode(
        nn, :grid,
        (q = ξ_grid[1:(reduced_dim ÷ 2)], p = ξ_grid[(reduced_dim ÷ 2 + 1):reduced_dim]))
    @test isapprox(F_grid_dec_nt, F_grid_dec; atol = 8 * eps(T), rtol = 8 * eps(T))

    # translate(nn, from, to, x) == decode(nn, to, encode(nn, from, x))
    @test translate(nn, :particles, :grid, x_part) ==
          decode(nn, :grid, encode(nn, :particles, x_part))
    @test (@inferred translate(nn, Val(:particles), Val(:grid), x_part)) ==
          translate(nn, :particles, :grid, x_part)
    @test translate(nn, :grid, :dlra, F_grid) ==
          decode(nn, :dlra, encode(nn, :grid, F_grid))
    @test translate(nn, :dlra, :spline, x_dlra) ==
          decode(nn, :spline, encode(nn, :dlra, x_dlra))

    # 2. Exact right-inversion encode(nn, mod, decode(nn, mod, ξ)) ≈ ξ to 50 eps(T)
    # Global right-invertible frontends on arbitrary ξ ∈ R^{2n}:
    for ξ_test in (
        randn(rng, T, reduced_dim),
        zeros(T, reduced_dim),
        fill(floatmin(T) * T(16), reduced_dim)
    )
        for mod in (:grid, :spline, :dlra, :particles_slot_cm)
            x_rec = decode(nn, mod, ξ_test)
            ξ_rt = encode(nn, mod, x_rec)
            @test eltype(ξ_rt) === T
            @test norm(ξ_rt - ξ_test) <= 50 * eps(T) * max(one(T), norm(ξ_test))
        end
    end

    # Batched latent matrix ξ_batch ∈ R^{2n × B} for :grid, :spline, :particles_slot_cm
    ξ_batch = randn(rng, T, reduced_dim, 3)
    for mod in (:grid, :spline, :particles_slot_cm)
        x_rec_b = decode(nn, mod, ξ_batch)
        ξ_rt_b = encode(nn, mod, x_rec_b)
        @test eltype(ξ_rt_b) === T && size(ξ_rt_b) == size(ξ_batch)
        @test norm(ξ_rt_b - ξ_batch) <= 50 * eps(T) * max(one(T), norm(ξ_batch))
    end

    # Particles with :sort and :slot (NSlots > 1) on the canonical Weyl chamber:
    core_ch = WeightTiedSymplecticAutoencoder(
        full_dim, 2;
        n_blocks = 1,
        n_layers = 0,
        n_ambient_layers = 0,
        sympnet_upscale = 2
    )
    arch_ch = MultiModalSymplecticAutoencoder(
        (particles = fe_sort, slot = fe_slot), core_ch)
    nn_ch = NeuralNetwork(arch_ch, CPU(), T; rng = rng)

    # Set core Stiefel leaf Φ_core ∈ St(1, 4) so decoded state for ξ_q > 0 lies in the
    # canonical chamber for both :sort (increasing particle row-1 q) and :slot (slot 2 active)
    Φ_core = zeros(T, N, 1)
    Φ_core[3, 1] = one(T)
    copyto!(parent(first(values(params(nn_ch).core)).weight), Φ_core)
    H_rel_3 = GeometricMachineLearning._helmert_relative_matrix(T, 3)
    Φ_rel_mat = reshape(transpose(H_rel_3) * T[-2.0, 0.8, 1.2], 2, 1)
    Φ_rel_mat ./= norm(Φ_rel_mat)
    copyto!(parent(params(nn_ch).frontends.slot.Φ_rel), Φ_rel_mat)

    ξ_slot_test = T[0.8, -0.5]
    x_slot_rec = @inferred decode(nn_ch, :slot, ξ_slot_test)
    ξ_slot_rt = @inferred encode(nn_ch, :slot, x_slot_rec)
    @test norm(ξ_slot_rt - ξ_slot_test) <= 50 * eps(T) * norm(ξ_slot_test)

    # For :sort (M = 2, d = 2), particle 1 has row-1 index 1 and particle 2 has row-1 index 3:
    # Φ_core[1, 1] = -1/sqrt(2), Φ_core[3, 1] = +1/sqrt(2) puts q_1 < q_2 for ξ_1 > 0
    Φ_core_sort = T[-inv(sqrt(T(2))), zero(T), inv(sqrt(T(2))), zero(T)]
    copyto!(parent(first(values(params(nn_ch).core)).weight), reshape(Φ_core_sort, N, 1))
    ξ_sort_test = T[0.9, -0.4]
    x_sort_rec = @inferred decode(nn_ch, :particles, ξ_sort_test)
    ξ_sort_rt = @inferred encode(nn_ch, :particles, x_sort_rec)
    @test norm(ξ_sort_rt - ξ_sort_test) <= 50 * eps(T) * norm(ξ_sort_test)
end

@testset "Clause 2: Multi-modal losses, zero-input edge case, M1, and Zygote vs FD gradients" begin
    @test isdefined(GeometricMachineLearning, :CharbonnierRelativeLoss)
    @test isdefined(GeometricMachineLearning, :IsometricRolloutLoss)
    @test isdefined(GeometricMachineLearning, :CrossModalCycleLoss)
    @test CharbonnierRelativeLoss <: NetworkLoss
    @test IsometricRolloutLoss <: NetworkLoss
    @test CrossModalCycleLoss <: NetworkLoss

    # Default keyword checks
    l_charb_def = CharbonnierRelativeLoss()
    @test l_charb_def.ε == 1e-3
    @test l_charb_def.λ_lat == 1e-2

    step_id = z -> z
    l_iso_def = IsometricRolloutLoss(step_id)
    @test l_iso_def.α_lat == 1.0
    @test l_iso_def.α_rec == 1.0
    @test l_iso_def.α_H == 0.1

    l_cyc_def = CrossModalCycleLoss(:particles, :grid)
    @test l_cyc_def.mod_a === :particles
    @test l_cyc_def.mod_b === :grid
    @test l_cyc_def.β_align == 1.0
    @test l_cyc_def.β_cycle == 0.1

    # Edge validation for loss hyperparameters
    re_charb_eps = r"^(?:ArgumentError: )?CharbonnierRelativeLoss requires ε > 0 and λ_lat >= 0, got ε = 0\.0, λ_lat = 0\.01\.$"
    re_charb_lam = r"^(?:ArgumentError: )?CharbonnierRelativeLoss requires ε > 0 and λ_lat >= 0, got ε = 0\.001, λ_lat = -0\.1\.$"
    @test_throws re_charb_eps CharbonnierRelativeLoss(; ε = 0.0)
    @test_throws re_charb_lam CharbonnierRelativeLoss(; λ_lat = -0.1)

    re_iso = r"^(?:ArgumentError: )?IsometricRolloutLoss requires non-negative weights α_lat, α_rec, α_H, got α_lat = -1\.0, α_rec = 1\.0, α_H = 0\.1\.$"
    @test_throws re_iso IsometricRolloutLoss(step_id; α_lat = -1.0)

    re_cyc = r"^(?:ArgumentError: )?CrossModalCycleLoss requires non-negative weights β_align and β_cycle, got β_align = 1\.0, β_cycle = -0\.1\.$"
    @test_throws re_cyc CrossModalCycleLoss(:particles, :grid; β_cycle = -0.1)

    @testset "Forward evaluation, @inferred, zero edge case, and M1 ($T)" for T in REAL_ELTYPES
        rng = Random.MersenneTwister(202)
        nx, nv = 4, 4
        mx, mv = 2, 1
        fe_part = CanonicalSlotLayer(3, 1, 2) # output dim = 4
        fe_grid = BilinearStiefelLayer(nx, nv, mx, mv) # output dim = 4
        core = WeightTiedSymplecticAutoencoder(
            4, 2; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)
        arch = MultiModalSymplecticAutoencoder(
            (particles = fe_part, grid = fe_grid), core)
        nn = NeuralNetwork(arch, CPU(), T; rng = rng)

        loss_c = CharbonnierRelativeLoss(; ε = T(1e-2), λ_lat = T(5e-2))
        F = randn(rng, T, nx, nv)
        F_hat = decode(nn, :grid, encode(nn, :grid, F))

        # Manual formula check
        z = encode(nn, :grid, F)
        z_hat = encode(nn, :grid, F_hat)
        eps_T = T(1e-2)
        lam_T = T(5e-2)
        expected_c = (sqrt(sum(abs2, F_hat .- F) + eps_T^2) - eps_T) /
                     sqrt(sum(abs2, F) + eps_T^2) +
                     lam_T * sum(abs2, z_hat .- z) / (sum(abs2, F) + eps_T^2)

        val_c1 = @inferred loss_c(nn, :grid, F)
        val_c2 = @inferred loss_c(nn, :grid, F_hat, F)
        @test eltype(val_c1) === T && eltype(val_c2) === T
        @test isapprox(val_c1, expected_c; atol = 10 * eps(T), rtol = 10 * eps(T))
        @test isapprox(val_c2, expected_c; atol = 10 * eps(T), rtol = 10 * eps(T))

        # Independent perturbation F_pert != F_hat so z_hat != z and the latent penalty term is active (catches O3)
        F_pert = F .+ T(0.5) .* ones(T, nx, nv)
        z_pert = encode(nn, :grid, F_pert)
        expected_c_pert = (sqrt(sum(abs2, F_pert .- F) + eps_T^2) - eps_T) /
                          sqrt(sum(abs2, F) + eps_T^2) +
                          lam_T * sum(abs2, z_pert .- z) / (sum(abs2, F) + eps_T^2)
        val_c_pert = @inferred loss_c(nn, :grid, F_pert, F)
        @test isapprox(val_c_pert, expected_c_pert; atol = 10 * eps(T), rtol = 10 * eps(T))

        # Mutant M1 check: when x̂ == x != 0, CharbonnierRelativeLoss MUST be exact zero(T)!
        # Omitting `- ε` in the numerator yields ε / sqrt(||x||_F^2 + ε^2) > 0.
        val_identical = @inferred loss_c(nn, :grid, F, F)
        @test val_identical === zero(T)

        # Decided at the edges: when x = 0 and x̂ = 0, returns exact zero(T) (not NaN)
        F_zero = zeros(T, nx, nv)
        val_zero = @inferred loss_c(nn, :grid, F_zero, F_zero)
        @test !isnan(val_zero) && val_zero === zero(T)

        # IsometricRolloutLoss forward & @inferred
        J2 = [zero(T) one(T); -one(T) zero(T)]
        dt = T(0.05)
        step_lin = ξ -> ξ .+ dt .* (J2 * ξ)
        H_lat = ξ -> T(0.5) * sum(abs2, ξ)
        loss_iso = IsometricRolloutLoss(
            step_lin; α_lat = T(1.0), α_rec = T(0.5), α_H = T(0.2), H = H_lat)

        re_len = r"^(?:ArgumentError: )?IsometricRolloutLoss requires at least 2 trajectory steps, got 1\.$"
        @test_throws re_len loss_iso(nn, :grid, [F])

        F_traj = [randn(rng, T, nx, nv) for _ in 1:4]
        val_iso = @inferred loss_iso(nn, :grid, F_traj)
        @test eltype(val_iso) === T && val_iso > zero(T)

        # Manual formula check for IsometricRolloutLoss
        z_enc = [encode(nn, :grid, F_k) for F_k in F_traj]
        z_roll = Vector{typeof(z_enc[1])}(undef, 4)
        z_roll[1] = z_enc[1]
        for k in 1:3
            z_roll[k + 1] = step_lin(z_roll[k])
        end
        err_lat = sum(sum(abs2, z_roll[k] .- z_enc[k]) for k in 2:4) / T(3)
        err_rec = sum(sum(abs2, decode(nn, :grid, z_roll[k]) .- F_traj[k]) for k in 2:4) /
                  T(3)
        err_H = sum(abs2(H_lat(z_roll[k]) - H_lat(z_roll[1])) for k in 2:4) / T(3)
        expected_iso = T(1.0) * err_lat + T(0.5) * err_rec + T(0.2) * err_H
        @test isapprox(val_iso, expected_iso; atol = 15 * eps(T), rtol = 15 * eps(T))

        # CrossModalCycleLoss forward & @inferred
        loss_cyc = CrossModalCycleLoss(
            :particles, :grid; β_align = T(1.2), β_cycle = T(0.3)
        )
        x_p = (q = randn(rng, T, 1, 3), p = randn(rng, T, 1, 3))
        val_cyc = @inferred loss_cyc(nn, x_p, F)
        @test eltype(val_cyc) === T && val_cyc > zero(T)

        z_a = encode(nn, :particles, x_p)
        z_b = encode(nn, :grid, F)
        z_a_cyc = encode(nn, :grid, decode(nn, :grid, z_a))
        expected_cyc = T(1.2) * sum(abs2, z_a .- z_b) + T(0.3) * sum(abs2, z_a_cyc .- z_a)
        @test isapprox(val_cyc, expected_cyc; atol = 10 * eps(T), rtol = 10 * eps(T))

        # Reverse direction CrossModalCycleLoss(:grid, :particles) where :particles is only chamber-wise right-invertible (catches O5)
        loss_cyc_rev = CrossModalCycleLoss(
            :grid, :particles; β_align = T(1.2), β_cycle = T(0.5))
        val_cyc_rev = @inferred loss_cyc_rev(nn, F, x_p)
        z_b_cyc = encode(nn, :particles, decode(nn, :particles, z_b))
        expected_cyc_rev = T(1.2) * sum(abs2, z_b .- z_a) +
                           T(0.5) * sum(abs2, z_b_cyc .- z_b)
        @test isapprox(
            val_cyc_rev, expected_cyc_rev; atol = 10 * eps(T), rtol = 10 *
                                                                      eps(T)
        )

        # Another precision than the network's raises an ArgumentError that names both.
        S = T === Float32 ? Float64 : Float32
        msg(name, what) = "$name: $what has element type $S, but the network has element type $T."
        loss_iso_int = IsometricRolloutLoss(ξ -> S.(step_lin(ξ)); H = H_lat)
        @test_throws msg("IsometricRolloutLoss", "the integrator's output") loss_iso_int(
            nn, :grid, F_traj)
        loss_iso_H = IsometricRolloutLoss(step_lin; H = ξ -> S(0.5) * S(sum(abs2, ξ)))
        @test_throws msg("IsometricRolloutLoss", "the value of H") loss_iso_H(
            nn, :grid, F_traj)
        @test_throws msg("IsometricRolloutLoss", "the data") loss_iso(
            nn, :grid, [S.(F_k) for F_k in F_traj])
        @test_throws msg("CrossModalCycleLoss", "x_b") loss_cyc(nn, x_p, S.(F))
    end

    @testset "Analytical Zygote gradients match central finite differences in Float64" begin
        rng = Random.MersenneTwister(303)
        T = Float64
        nx, nv = 4, 4
        mx, mv = 2, 1
        # Full dim = 2 * 2 * 1 = 4, reduced_dim = 2
        fe_part = CanonicalSlotLayer(3, 1, 2) # M = 3, d = 1, NSlots = 2 -> NOut = 4
        fe_grid = BilinearStiefelLayer(nx, nv, mx, mv) # NOut = 4
        core = WeightTiedSymplecticAutoencoder(
            4, 2; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)
        arch = MultiModalSymplecticAutoencoder((particles = fe_part, grid = fe_grid), core)
        nn = NeuralNetwork(arch, CPU(), T; rng = rng)

        F = randn(rng, T, nx, nv)
        F_traj = [randn(rng, T, nx, nv) for _ in 1:3]
        # Particle state with well-separated q[1, :] and relative mode norms so sorting permutations are locally constant
        x_p = (q = T[-2.0 0.1 2.5], p = T[0.4 -1.3 0.9])

        loss_c = CharbonnierRelativeLoss(; ε = 1e-2, λ_lat = 5e-2)
        F_zero = zeros(T, nx, nv)
        g_zero = Zygote.gradient(x_hat -> loss_c(nn, :grid, x_hat, F_zero), F_zero)[1]
        @test all(isfinite, g_zero) && all(iszero, g_zero)
        g_same = Zygote.gradient(x_hat -> loss_c(nn, :grid, x_hat, F), F)[1]
        @test all(isfinite, g_same) && all(iszero, g_same)

        step_fn = ξ -> [ξ[1] + 0.1 * ξ[2] + 0.05 * tanh(ξ[2]), ξ[2] - 0.1 * ξ[1]]
        H_fn = ξ -> 0.5 * sum(abs2, ξ) + 0.05 * sum(ξ .^ 4)
        loss_iso = IsometricRolloutLoss(
            step_fn; α_lat = 1.0, α_rec = 0.8, α_H = 0.25, H = H_fn)
        loss_cyc = CrossModalCycleLoss(:particles, :grid; β_align = 1.0, β_cycle = 0.2)

        flat0, rebuild_ps = _flatten_params(nn.params)
        h = 1e-6
        tol = 10 * sqrt(eps(Float64))

        for (name, f_ps) in (
            ("CharbonnierRelativeLoss", ps -> loss_c(nn.model, ps, :grid, F)),
            ("IsometricRolloutLoss", ps -> loss_iso(nn.model, ps, :grid, F_traj)),
            ("CrossModalCycleLoss", ps -> loss_cyc(nn.model, ps, x_p, F))
        )
            g_zyg = Zygote.gradient(f_ps, nn.params)[1]
            g_vec = _extract_grad_vec(g_zyg)
            @test length(g_vec) == length(flat0)

            g_fd = zeros(Float64, length(flat0))
            w_plus = copy(flat0)
            w_minus = copy(flat0)
            for i in eachindex(flat0)
                w_plus[i] = flat0[i] + h
                w_minus[i] = flat0[i] - h
                g_fd[i] = (f_ps(rebuild_ps(w_plus)) - f_ps(rebuild_ps(w_minus))) / (2 * h)
                w_plus[i] = flat0[i]
                w_minus[i] = flat0[i]
            end
            @test isapprox(g_vec, g_fd; atol = tol, rtol = tol)
        end
    end

    @testset "Chain particle frontends, SymmetricSymplecticAutoencoder core, and batched @inferred" begin
        rng = Random.MersenneTwister(404)
        T = Float64
        nx, nv, mx, mv, B = 6, 6, 2, 1, 3
        _grad_vec(x::AbstractArray) = vec(Float64.(parent(x)))
        _grad_vec(::Nothing) = Float64[]
        _grad_vec(nt) = isempty(nt) ? Float64[] : mapreduce(_grad_vec, vcat, values(nt))
        fe_pw_chain = Chain(
            PairwiseSymplecticLayerQ(2, 1, 4), PairwiseSymplecticLayerP(2, 1, 4))
        fe_grid = BilinearStiefelLayer(nx, nv, mx, mv)
        fe_dlra = LowRankStiefelLayer(nx, nv, 2, 1; r_out = 2)
        core_id = WeightTiedSymplecticAutoencoder(
            4, 4; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)
        nn_pw = NeuralNetwork(
            MultiModalSymplecticAutoencoder(
                (particle = fe_pw_chain, grid = fe_grid, dlra = fe_dlra), core_id),
            CPU(), T; rng = rng)
        @test params(nn_pw).frontends.particle isa NamedTuple
        x_qp2 = (q = randn(rng, T, 1, 2), p = randn(rng, T, 1, 2))
        x_vec2 = vcat(vec(x_qp2.q), vec(x_qp2.p))
        z_pw_qp = @inferred encode(nn_pw, :particle, x_qp2)
        z_pw_vec = @inferred encode(nn_pw, :particle, x_vec2)
        @test isapprox(z_pw_qp, z_pw_vec; atol = 10 * eps(T), rtol = 10 * eps(T))
        x_pw_rec = @inferred decode(nn_pw, Val(:particle), z_pw_qp)
        @test isapprox(x_pw_rec, x_vec2; atol = 50 * eps(T), rtol = 50 * eps(T))
        core_sa = SymplecticAutoencoder(
            4, 2; n_encoder_blocks = 1, n_decoder_blocks = 1,
            n_encoder_layers = 1, n_decoder_layers = 1,
            n_decoder_output_layers = 1, sympnet_upscale = 2)
        nn_sa_part = NeuralNetwork(
            MultiModalSymplecticAutoencoder((particle = fe_pw_chain,), core_sa),
            CPU(), T; rng = rng)
        z_sa_vec = @inferred encode(nn_sa_part, :particle, x_vec2)
        @test z_sa_vec isa Vector{T} && length(z_sa_vec) == 2
        @test (@inferred encode(nn_sa_part, :particle, randn(rng, T, 4, 3))) isa Matrix{T}
        @test (@inferred decode(nn_sa_part, :particle, z_sa_vec)) isa Vector{T}
        @test (@inferred translate(nn_sa_part, :particle, :particle, x_vec2)) isa Vector{T}
        fe_can_chain = Chain(
            CanonicalSortLayer(3, 1, 1), PairwiseSymplecticLayerQ(3, 1, 4),
            PairwiseSymplecticLayerP(3, 1, 4), CanonicalSlotLayer(3, 1, 2))
        core_wt = WeightTiedSymplecticAutoencoder(
            4, 2; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)
        nn_can = NeuralNetwork(
            MultiModalSymplecticAutoencoder(
                (particle = fe_can_chain, grid = fe_grid, dlra = fe_dlra), core_wt),
            CPU(), T; rng = rng)
        @test params(nn_can).frontends.particle isa NamedTuple
        x_qp3 = (q = T[-2.0 0.1 2.5], p = T[0.4 -1.3 0.9])
        x_vec3 = vcat(vec(x_qp3.q), vec(x_qp3.p))
        z_can_qp = @inferred encode(nn_can, :particle, x_qp3)
        z_can_vec = @inferred encode(nn_can, :particle, x_vec3)
        @test isapprox(z_can_qp, z_can_vec; atol = 10 * eps(T), rtol = 10 * eps(T))
        x_can_rec = @inferred decode(nn_can, Val(:particle), z_can_qp)
        @test length(x_can_rec) == 6 &&
              isapprox(x_can_rec, decode(nn_can, :particle, z_can_qp); atol = 10 * eps(T))
        fe_can_d2 = Chain(
            CanonicalSortLayer(3, 2, 1), PairwiseSymplecticLayerQ(3, 2, 4),
            PairwiseSymplecticLayerP(3, 2, 4), CanonicalSlotLayer(3, 2, 2))
        core_wt_d2 = WeightTiedSymplecticAutoencoder(
            8, 4; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)
        nn_d2 = NeuralNetwork(
            MultiModalSymplecticAutoencoder((particle = fe_can_d2,), core_wt_d2),
            CPU(), T; rng = rng)
        x_qp_d2 = (
            q = T[3.0 -1.0 2.0; -2.0 4.0 0.5], p = T[0.2 -0.5 0.8; 1.1 -0.3 0.4])
        x_vec_d2 = vcat(vec(x_qp_d2.q), vec(x_qp_d2.p))
        ps_d2 = params(nn_d2).frontends.particle
        s1_d2 = fe_can_d2.layers[1](x_qp_d2, ps_d2.L1)
        s2_d2 = fe_can_d2.layers[2]((q = s1_d2.q, p = s1_d2.p), ps_d2.L2)
        s4_d2 = fe_can_d2.layers[4](fe_can_d2.layers[3](s2_d2, ps_d2.L3), ps_d2.L4)
        z_exp_d2 = encoder(nn_d2)(vcat(vec(s4_d2.q), vec(s4_d2.p)))
        tol = 10 * eps(T)
        @test isapprox(@inferred(encode(nn_d2, :particle, x_qp_d2)), z_exp_d2; atol = tol)
        @test isapprox(@inferred(encode(nn_d2, :particle, x_vec_d2)), z_exp_d2; atol = tol)
        x_qp3_batch = (q = randn(rng, T, 1, 3, B), p = randn(rng, T, 1, 3, B))
        x_mat3_batch = vcat(reshape(x_qp3_batch.q, 3, B), reshape(x_qp3_batch.p, 3, B))
        F_batch = randn(rng, T, nx, nv, B)
        x_dlra = (
            X = Matrix(qr(randn(rng, T, nx, 2)).Q),
            S = randn(rng, T, 2, 2),
            V = Matrix(qr(randn(rng, T, nv, 2)).Q))
        X_b, S_b, V_b = zeros(T, nx, 2, B), randn(rng, T, 2, 2, B), zeros(T, nv, 2, B)
        for b in 1:B
            X_b[:, :, b] .= Matrix(qr(randn(rng, T, nx, 2)).Q)
            V_b[:, :, b] .= Matrix(qr(randn(rng, T, nv, 2)).Q)
        end
        x_dlra_batch = (X = X_b, S = S_b, V = V_b)
        z_b_qp = @inferred encode(nn_can, :particle, x_qp3_batch)
        z_b_mat = @inferred encode(nn_can, Val(:particle), x_mat3_batch)
        @test size(z_b_qp) == (2, B) && isapprox(z_b_qp, z_b_mat; atol = tol)
        @test size(@inferred decode(nn_can, Val(:particle), z_b_qp)) == (6, B)
        z_b_grid = @inferred encode(nn_can, :grid, F_batch)
        @test size(z_b_grid) == (2, B)
        @test size(@inferred decode(nn_can, Val(:grid), z_b_grid)) == (nx, nv, B)
        z_dlra_1 = @inferred encode(nn_can, :dlra, x_dlra)
        z_b_dlra = @inferred encode(nn_can, :dlra, x_dlra_batch)
        dec_b_dlra = @inferred decode(nn_can, Val(:dlra), z_b_dlra)
        @test length(z_dlra_1) == 2 && size(z_b_dlra) == (2, B) &&
              size(dec_b_dlra.X) == (nx, 2, B)
        F_from_p_b = @inferred translate(nn_can, Val(:particle), Val(:grid), x_mat3_batch)
        @test size(F_from_p_b) == (nx, nv, B)
        loss_c = CharbonnierRelativeLoss(; ε = 1e-2, λ_lat = 5e-2)
        @test (@inferred loss_c(nn_can, :particle, x_qp3)) > zero(T)
        @test (@inferred loss_c(nn_can, :dlra, x_dlra)) > zero(T)
        @test isapprox(GeometricMachineLearning._frob_sq_norm(x_dlra_batch),
            sum(b -> sum(abs2, X_b[:, :, b] * S_b[:, :, b] * V_b[:, :, b]'), 1:B);
            rtol = 8 * eps(T))
        @test loss_c(nn_can, :dlra, x_dlra_batch) > zero(T)
        calls_mm, calls_fe, calls_core = Ref(0), Ref(0), Ref(0)
        NeuralNetwork(
            MultiModalSymplecticAutoencoder((grid = fe_grid,), core_wt), CPU(), T;
            rng = rng, initializer = CountingInitializer(calls_mm))
        initialparameters(rng, CountingInitializer(calls_fe), fe_grid, CPU(), T)
        NeuralNetwork(
            core_wt, CPU(), T; rng = rng, initializer = CountingInitializer(calls_core))
        @test calls_core[] > 0 && calls_mm[] == calls_fe[] + calls_core[]
        g_enc_pw = GeometricMachineLearning.Zygote.gradient(
            ps -> sum(encode(nn_pw.model, ps, :particle, x_qp2)), params(nn_pw))[1]
        g_dec_pw = GeometricMachineLearning.Zygote.gradient(
            ps -> sum(decode(nn_pw.model, ps, :particle, z_pw_qp)), params(nn_pw))[1]
        x_dlra_hat = @inferred decode(nn_can, Val(:dlra), z_dlra_1)
        g_dlra = GeometricMachineLearning.Zygote.gradient(
            ps -> loss_c(nn_can.model, ps, :dlra, x_dlra_hat, x_dlra), params(nn_can))[1]
        @test all(isfinite, _grad_vec(g_enc_pw)) && all(isfinite, _grad_vec(g_dec_pw)) &&
              all(isfinite, _grad_vec(g_dlra))
        core_sym = SymmetricSymplecticAutoencoder(
            2, 1, 2; n_pairwise_layers = 1, pairwise_upscale = 2,
            n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)
        nn_sym = NeuralNetwork(
            MultiModalSymplecticAutoencoder(
                (particle = fe_can_chain, grid = fe_grid), core_sym),
            CPU(), T; rng = rng)
        F_single = randn(rng, T, nx, nv)
        z_sym_p = @inferred encode(nn_sym, :particle, x_qp3)
        z_sym_g = @inferred encode(nn_sym, Val(:grid), F_single)
        F_sym_rec = @inferred decode(nn_sym, Val(:grid), z_sym_g)
        F_sym_tr = @inferred translate(nn_sym, Val(:particle), Val(:grid), x_qp3)
        @test length(z_sym_p) == 2 && size(F_sym_rec) == size(F_sym_tr) == (nx, nv)
        g_sym = GeometricMachineLearning.Zygote.gradient(
            ps -> sum(encode(nn_sym.model, ps, :grid, F_single)), params(nn_sym))[1]
        @test all(isfinite, _grad_vec(g_sym))
    end
end
