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

@testset "MultiModalSymplecticAutoencoder construction and edge cases" begin
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

@testset "encode, decode, translate and right-invertible round-trip ($T)" for T in REAL_ELTYPES
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

@testset "Multi-modal losses, zero-input edge case, exact-zero loss, and Zygote vs FD gradients" begin
    @test isdefined(GeometricMachineLearning, :CharbonnierRelativeLoss)
    @test isdefined(GeometricMachineLearning, :IsometricRolloutLoss)
    @test isdefined(GeometricMachineLearning, :CrossModalLoss)
    @test CharbonnierRelativeLoss <: NetworkLoss
    @test IsometricRolloutLoss <: NetworkLoss
    @test CrossModalLoss <: NetworkLoss

    # Default keyword checks
    l_charb_def = CharbonnierRelativeLoss()
    @test l_charb_def.ε == 1e-3
    @test l_charb_def.λ_lat == 1e-2

    step_id = z -> z
    H_lat_def = z -> sum(abs2, z) / 2
    l_iso_def = IsometricRolloutLoss(step_id)
    @test l_iso_def.α_lat == 1.0
    @test l_iso_def.α_rec == 1.0
    @test l_iso_def.α_H == 0.0
    @test l_iso_def.H === nothing
    @test IsometricRolloutLoss(step_id; α_H = 0.1, H = H_lat_def).H === H_lat_def

    # There is no default latent Hamiltonian: α_H > 0 requires H, and α_H = 0 needs none.
    re_iso_H = r"^(?:ArgumentError: )?IsometricRolloutLoss requires a latent Hamiltonian H when α_H > 0, got α_H = 0\.1 and H = nothing\. Pass H, or set α_H = 0\.$"
    @test_throws re_iso_H IsometricRolloutLoss(step_id; α_H = 0.1)
    @test_throws re_iso_H IsometricRolloutLoss(step_id; α_H = 0.1, H = nothing)
    @test IsometricRolloutLoss(step_id; α_H = 0.0).H === nothing
    # With α_H = 0 a given H is not stored, so it is never called.
    @test IsometricRolloutLoss(step_id; α_H = 0.0, H = H_lat_def).H === nothing

    l_cyc_def = CrossModalLoss(:particles, :grid)
    @test l_cyc_def.mod_a === :particles
    @test l_cyc_def.mod_b === :grid
    @test l_cyc_def.β_align == 1.0
    @test l_cyc_def.β_cross == 1.0
    @test !hasfield(typeof(l_cyc_def), :β_cycle)
    @test_throws MethodError CrossModalLoss(:particles, :grid; β_cycle = 0.1)

    # Edge validation for loss hyperparameters
    re_charb_eps = r"^(?:ArgumentError: )?CharbonnierRelativeLoss requires ε > 0 and λ_lat >= 0, got ε = 0\.0, λ_lat = 0\.01\.$"
    re_charb_lam = r"^(?:ArgumentError: )?CharbonnierRelativeLoss requires ε > 0 and λ_lat >= 0, got ε = 0\.001, λ_lat = -0\.1\.$"
    @test_throws re_charb_eps CharbonnierRelativeLoss(; ε = 0.0)
    @test_throws re_charb_lam CharbonnierRelativeLoss(; λ_lat = -0.1)

    re_iso = r"^(?:ArgumentError: )?IsometricRolloutLoss requires non-negative weights α_lat, α_rec, α_H, got α_lat = -1\.0, α_rec = 1\.0, α_H = 0\.0\.$"
    @test_throws re_iso IsometricRolloutLoss(step_id; α_lat = -1.0)

    re_cross = r"^(?:ArgumentError: )?CrossModalLoss requires non-negative weights β_align and β_cross, got β_align = 1\.0, β_cross = -0\.1\.$"
    @test_throws re_cross CrossModalLoss(:particles, :grid; β_cross = -0.1)
    re_cross_nan = r"^(?:ArgumentError: )?CrossModalLoss requires non-negative weights β_align and β_cross, got β_align = 1\.0, β_cross = NaN\.$"
    @test_throws re_cross_nan CrossModalLoss(:particles, :grid; β_cross = NaN)
    re_align = r"^(?:ArgumentError: )?CrossModalLoss requires non-negative weights β_align and β_cross, got β_align = -0\.1, β_cross = 1\.0\.$"
    @test_throws re_align CrossModalLoss(:particles, :grid; β_align = -0.1)

    @testset "Forward evaluation, @inferred, zero edge case, and exact-zero loss ($T)" for T in REAL_ELTYPES
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
        expected_c_rec = (sqrt(sum(abs2, F_hat .- F) + eps_T^2) - eps_T) /
                         sqrt(sum(abs2, F) + eps_T^2)
        expected_c = expected_c_rec +
                     lam_T * sum(abs2, z_hat .- z) / (sum(abs2, F) + eps_T^2)

        # Both forms have the latent term; the self-reconstruction form uses x̂ = decode(encode(x)).
        val_c1 = @inferred loss_c(nn, :grid, F)
        val_c2 = @inferred loss_c(nn, :grid, F_hat, F)
        @test eltype(val_c1) === T && eltype(val_c2) === T
        @test isapprox(val_c1, expected_c; atol = 10 * eps(T), rtol = 10 * eps(T))
        @test isapprox(val_c2, expected_c; atol = 10 * eps(T), rtol = 10 * eps(T))

        # In the self-reconstruction form, a pre-split field (q = F⁺, p = F⁻) gives the value of
        # the full field F: the decoded field is compared with F⁺ and F⁻ recombined.
        F_plus = (F[:, 1:(nv ÷ 2)] .+ F[:, nv:-1:(nv ÷ 2 + 1)]) ./ sqrt(T(2))
        F_minus = (F[:, 1:(nv ÷ 2)] .- F[:, nv:-1:(nv ÷ 2 + 1)]) ./ sqrt(T(2))
        F_split = (q = F_plus, p = F_minus)
        @test isapprox(@inferred(loss_c(nn, :grid, F_split)), val_c1;
            atol = 10 * eps(T), rtol = 10 * eps(T))
        # The form with an external x̂ recombines a pre-split x and a pre-split x̂ in the same way.
        @test isapprox(@inferred(loss_c(nn, :grid, F_hat, F_split)), val_c2;
            atol = 10 * eps(T), rtol = 10 * eps(T))
        @test @inferred(loss_c(nn, :grid, F_split, F_split)) === zero(T)
        # The two halves of a pre-split pair have one element type: halves of different element
        # types raise an ArgumentError in both forms, and a pair of one element type gives the
        # value of the full field that it recombines to by hand.
        if T === Float64
            F_split_mixed = (q = F_plus, p = Float32.(F_minus))
            re_mixed = r"^(?:ArgumentError: )?A pre-split pair \(q = F⁺, p = F⁻\) needs one element type, got q of element type Float64 and p of element type Float32\.$"
            @test_throws re_mixed loss_c(nn, :grid, F_hat, F_split_mixed)
            @test_throws re_mixed loss_c(nn, :grid, F_split_mixed)
            F_minus_same = Float64.(F_split_mixed.p)
            F_split_same = (q = F_plus, p = F_minus_same)
            F_same_full = hcat((F_plus .+ F_minus_same) ./ sqrt(T(2)),
                ((F_plus .- F_minus_same) ./ sqrt(T(2)))[:, end:-1:1])
            val_same = @inferred loss_c(nn, :grid, F_hat, F_split_same)
            val_same_full = loss_c(nn, :grid, F_hat, F_same_full)
            @test isapprox(val_same, val_same_full; rtol = 10 * eps(T))
        end

        # Independent perturbation F_pert != F_hat so z_hat != z and the latent penalty term is active (catches a dropped latent penalty)
        F_pert = F .+ T(0.5) .* ones(T, nx, nv)
        z_pert = encode(nn, :grid, F_pert)
        expected_c_pert = (sqrt(sum(abs2, F_pert .- F) + eps_T^2) - eps_T) /
                          sqrt(sum(abs2, F) + eps_T^2) +
                          lam_T * sum(abs2, z_pert .- z) / (sum(abs2, F) + eps_T^2)
        val_c_pert = @inferred loss_c(nn, :grid, F_pert, F)
        @test isapprox(val_c_pert, expected_c_pert; atol = 10 * eps(T), rtol = 10 * eps(T))

        # Exact-zero check: when x̂ == x != 0, CharbonnierRelativeLoss MUST be exact zero(T)!
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

        # With α_H = 0 and no H, the loss is the α_lat and α_rec terms alone.
        loss_iso_noH = IsometricRolloutLoss(step_lin; α_lat = T(1.0), α_rec = T(0.5), α_H = 0)
        val_iso_noH = @inferred loss_iso_noH(nn, :grid, F_traj)
        @test eltype(val_iso_noH) === T
        @test isapprox(val_iso_noH, T(1.0) * err_lat + T(0.5) * err_rec;
            atol = 15 * eps(T), rtol = 15 * eps(T))

        # IsometricRolloutLoss(integrator) has the weights α_lat = α_rec = 1 and no energy term.
        loss_iso_def = IsometricRolloutLoss(step_lin)
        val_iso_def = @inferred loss_iso_def(nn, :grid, F_traj)
        @test eltype(val_iso_def) === T
        @test isapprox(val_iso_def, err_lat + err_rec;
            atol = 15 * eps(T), rtol = 15 * eps(T))

        # A trajectory of pre-split fields gives the value of the trajectory of full fields.
        split_traj = [(
                          q = (F_k[:, 1:(nv ÷ 2)] .+ F_k[:, nv:-1:(nv ÷ 2 + 1)]) ./
                              sqrt(T(2)),
                          p = (F_k[:, 1:(nv ÷ 2)] .- F_k[:, nv:-1:(nv ÷ 2 + 1)]) ./
                              sqrt(T(2)))
                      for F_k in F_traj]
        @test isapprox(@inferred(loss_iso(nn, :grid, split_traj)), val_iso;
            atol = 15 * eps(T), rtol = 15 * eps(T))

        # CrossModalLoss forward & @inferred: the alignment term and the
        # cross-reconstruction term ‖decode(b, z_a) - x_b‖².
        loss_cyc = CrossModalLoss(
            :particles, :grid; β_align = T(1.2), β_cross = T(0.3)
        )
        x_p = (q = randn(rng, T, 1, 3), p = randn(rng, T, 1, 3))
        val_cyc = @inferred loss_cyc(nn, x_p, F)
        @test eltype(val_cyc) === T && val_cyc > zero(T)

        z_a = encode(nn, :particles, x_p)
        z_b = encode(nn, :grid, F)
        expected_cyc = T(1.2) * sum(abs2, z_a .- z_b) +
                       T(0.3) * sum(abs2, decode(nn, :grid, z_a) .- F)
        @test isapprox(val_cyc, expected_cyc; atol = 10 * eps(T), rtol = 10 * eps(T))

        # Reverse direction CrossModalLoss(:grid, :particles): the particle decoder returns
        # the flat state [vec(q); vec(p)].
        loss_cyc_rev = CrossModalLoss(
            :grid, :particles; β_align = T(1.2), β_cross = T(0.5))
        val_cyc_rev = @inferred loss_cyc_rev(nn, F, x_p)
        expected_cyc_rev = T(1.2) * sum(abs2, z_b .- z_a) +
                           T(0.5) * sum(
            abs2, decode(nn, :particles, z_b) .- vcat(vec(x_p.q), vec(x_p.p)))
        @test isapprox(
            val_cyc_rev, expected_cyc_rev; atol = 10 * eps(T), rtol = 10 * eps(T))

        # The three receivers, each with x_a and x_b and with the pair (x_a, x_b).
        ps_nt, ps_np = params(nn), nn.params
        @test (@inferred loss_cyc(nn, (x_p, F))) == val_cyc
        @test (@inferred loss_cyc(nn.model, ps_nt, x_p, F)) == val_cyc
        @test (@inferred loss_cyc(nn.model, ps_nt, (x_p, F))) == val_cyc
        @test (@inferred loss_cyc(nn.model, ps_np, x_p, F)) == val_cyc
        @test (@inferred loss_cyc(nn.model, ps_np, (x_p, F))) == val_cyc

        # A batch is the sum over its samples, with no division by B.
        B = 3
        x_p_b = (q = randn(rng, T, 1, 3, B), p = randn(rng, T, 1, 3, B))
        F_b = randn(rng, T, nx, nv, B)
        val_cyc_b = @inferred loss_cyc(nn, x_p_b, F_b)
        @test eltype(val_cyc_b) === T
        val_cyc_sum = sum(
            b -> loss_cyc(nn, (q = x_p_b.q[:, :, b], p = x_p_b.p[:, :, b]), F_b[:, :, b]),
            1:B)
        @test isapprox(val_cyc_b, val_cyc_sum; rtol = 10 * eps(T))
        @test (@inferred loss_cyc(nn, (x_p_b, F_b))) == val_cyc_b

        # x_a and x_b with different numbers of samples raise a DimensionMismatch.
        re_samples = r"^(?:DimensionMismatch: )?CrossModalLoss: the latent states of x_a and x_b have different sizes, \(2,\) and \(2, 3\)\.$"
        @test_throws re_samples loss_cyc(nn, x_p, F_b)
        re_samples_rev = r"^(?:DimensionMismatch: )?CrossModalLoss: the latent states of x_a and x_b have different sizes, \(2, 3\) and \(2,\)\.$"
        @test_throws re_samples_rev loss_cyc(nn, x_p_b, F)

        # A pre-split field (q = F⁺, p = F⁻) for x_b gives the value of the full field F. The
        # parity components come from explicit parity matrices: vec(F⁺) = P₊ vec(F).
        half = nv ÷ 2
        P_plus = zeros(T, nx * half, nx * nv)
        P_minus = zeros(T, nx * half, nx * nv)
        for j in 1:half, i in 1:nx

            row = (j - 1) * nx + i
            col_top = (j - 1) * nx + i
            col_bot = (nv - j) * nx + i
            P_plus[row, col_top] = P_plus[row, col_bot] = inv(sqrt(T(2)))
            P_minus[row, col_top] = inv(sqrt(T(2)))
            P_minus[row, col_bot] = -inv(sqrt(T(2)))
        end
        split_of(F2) = (q = reshape(P_plus * vec(F2), nx, half),
            p = reshape(P_minus * vec(F2), nx, half))
        @test isapprox(@inferred(loss_cyc(nn, x_p, split_of(F))), val_cyc;
            rtol = 10 * eps(T))
        F_b_split = (q = cat((split_of(F_b[:, :, b]).q for b in 1:B)...; dims = 3),
            p = cat((split_of(F_b[:, :, b]).p for b in 1:B)...; dims = 3))
        @test isapprox(@inferred(loss_cyc(nn, x_p_b, F_b_split)), val_cyc_b;
            rtol = 10 * eps(T))

        # Another precision than the network's raises an ArgumentError that names both.
        S = T === Float32 ? Float64 : Float32
        msg(name, what) = "$name: $what has element type $S, but the network has element type $T."
        loss_iso_int = IsometricRolloutLoss(ξ -> S.(step_lin(ξ)); α_H = 0.1, H = H_lat)
        @test_throws msg("IsometricRolloutLoss", "the integrator's output") loss_iso_int(
            nn, :grid, F_traj)
        loss_iso_H = IsometricRolloutLoss(
            step_lin; α_H = 0.1, H = ξ -> S(0.5) * S(sum(abs2, ξ)))
        @test_throws msg("IsometricRolloutLoss", "the value of H") loss_iso_H(
            nn, :grid, F_traj)
        @test_throws msg("IsometricRolloutLoss", "the data") loss_iso(
            nn, :grid, [S.(F_k) for F_k in F_traj])
        @test_throws msg("CrossModalLoss", "x_b") loss_cyc(nn, x_p, S.(F))

        # A particle state with one component of another precision than the network's raises
        # the same ArgumentError, whichever component it is.
        x_p_qS = (q = x_p.q, p = S.(x_p.p))
        x_p_pS = (q = S.(x_p.q), p = x_p.p)
        for x_mixed in (x_p_qS, x_p_pS)
            @test_throws msg("IsometricRolloutLoss", "the data") loss_iso(
                nn, :particles, [x_mixed, x_p])
            @test_throws msg("IsometricRolloutLoss", "the data") loss_iso(
                nn, :particles, [x_p, x_mixed])
            @test_throws msg("CrossModalLoss", "x_a") loss_cyc(nn, (x_mixed, F))
            @test_throws msg("CrossModalLoss", "x_b") loss_cyc_rev(nn, (F, x_mixed))
        end
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
        loss_cyc = CrossModalLoss(:particles, :grid; β_align = 1.0, β_cross = 0.2)

        flat0, rebuild_ps = _flatten_params(nn.params)
        h = 1e-6
        tol = 10 * sqrt(eps(Float64))

        for (name, f_ps) in (
            ("CharbonnierRelativeLoss", ps -> loss_c(nn.model, ps, :grid, F)),
            ("IsometricRolloutLoss", ps -> loss_iso(nn.model, ps, :grid, F_traj)),
            ("CrossModalLoss", ps -> loss_cyc(nn.model, ps, x_p, F))
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
        # Factors that already have the element type of the batch are reused, not modified:
        # value and gradient agree with the sum over the slices, which takes the 2D path.
        for Tb in (Float32, Float64)
            Xb, Sb, Vb = randn(rng, Tb, nx, 2, B), randn(rng, Tb, 2, 2, B),
            randn(rng, Tb, nv, 2, B)
            Xb0, Sb0, Vb0 = copy(Xb), copy(Sb), copy(Vb)
            slice(b, X, S, V) = (X = X[:, :, b], S = S[:, :, b], V = V[:, :, b])
            f_batch(X, S, V) = GeometricMachineLearning._frob_sq_norm((X = X, S = S, V = V))
            f_slices(X, S, V) = sum(
                b -> GeometricMachineLearning._frob_sq_norm(slice(b, X, S, V)), 1:B)
            val_b = f_batch(Xb, Sb, Vb)
            @test val_b isa Tb
            @test isapprox(val_b, f_slices(Xb, Sb, Vb); rtol = 32 * eps(Tb))
            g_b = Zygote.gradient(f_batch, Xb, Sb, Vb)
            g_s = Zygote.gradient(f_slices, Xb, Sb, Vb)
            @test all(isapprox.(g_b, g_s; rtol = 64 * eps(Tb)))
            @test Xb == Xb0 && Sb == Sb0 && Vb == Vb0
        end
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

# The untied core of the control networks below.
function _untied_core()
    SymplecticAutoencoder(
        4, 2; n_encoder_blocks = 1, n_decoder_blocks = 1, n_encoder_layers = 1,
        n_decoder_layers = 1, n_decoder_output_layers = 1, sympnet_upscale = 2)
end

@testset "Cross-reconstruction term against a dense linear oracle ($T)" for T in REAL_ELTYPES
    rng = Random.MersenneTwister(505)
    Nx, Nv, mx, mv = 4, 4, 2, 1
    half = Nv ÷ 2
    s = inv(sqrt(T(2)))
    # Exactly orthonormal leaves, with entries 0, ±1 and ±1/√2.
    Φx = T[s 0; -s 0; 0 1; 0 0]
    Φv = reshape(T[s, s], half, mv)
    Φc = reshape(T[s, -s], 2, 1)

    fe = BilinearStiefelLayer(Nx, Nv, mx, mv)
    core = WeightTiedSymplecticAutoencoder(
        4, 2; n_blocks = 1, n_layers = 0, n_ambient_layers = 0, sympnet_upscale = 2)
    nn = NeuralNetwork(
        MultiModalSymplecticAutoencoder((a = fe, b = fe), core), CPU(), T; rng = rng)
    nn_sa = NeuralNetwork(
        MultiModalSymplecticAutoencoder((a = fe, b = fe), _untied_core()), CPU(), T;
        rng = rng)
    for net in (nn, nn_sa), m in (:a, :b)

        copyto!(parent(getfield(params(net).frontends, m).Φx), Φx)
        copyto!(parent(getfield(params(net).frontends, m).Φv), Φv)
    end
    @test length(params(nn).core) == 1
    copyto!(parent(params(nn).core.L1.weight), Φc)

    # Dense encoder and decoder from the leaves and explicit parity matrices, vec(F) x-fastest:
    # vec(F⁺) = P₊ vec(F), vec(F⁻) = P₋ vec(F), vec(Φxᵀ G Φv) = kron(Φv, Φx)ᵀ vec(G), and the
    # weight-tied core with no SympNet layers is (q, p) ↦ (Φcᵀ q, Φcᵀ p).
    P_plus = zeros(T, Nx * half, Nx * Nv)
    P_minus = zeros(T, Nx * half, Nx * Nv)
    for j in 1:half, i in 1:Nx

        row = (j - 1) * Nx + i
        col_top = (j - 1) * Nx + i
        col_bot = (Nv - j) * Nx + i
        P_plus[row, col_top] = P_plus[row, col_bot] = s
        P_minus[row, col_top] = s
        P_minus[row, col_bot] = -s
    end
    K = kron(Φv, Φx)
    O = zeros(T, 1, 2)
    C_enc = [transpose(Φc) O; O transpose(Φc)]
    Enc = C_enc * vcat(transpose(K) * P_plus, transpose(K) * P_minus)
    Dec = hcat(transpose(P_plus) * K, transpose(P_minus) * K) * transpose(C_enc)

    x_a = T(3.7) .* randn(rng, T, Nx, Nv)
    x_b = T(3.7) .* randn(rng, T, Nx, Nv)
    z_a = Enc * vec(x_a)
    z_b = Enc * vec(x_b)
    @test isapprox(encode(nn, :a, x_a), z_a; rtol = 10 * eps(T))
    @test isapprox(vec(decode(nn, :b, z_a)), Dec * z_a; rtol = 10 * eps(T))

    β_align, β_cross = T(0.7), T(1.3)
    loss = CrossModalLoss(:a, :b; β_align = β_align, β_cross = β_cross)
    expected = β_align * sum(abs2, z_a - z_b) + β_cross * sum(abs2, Dec * z_a - vec(x_b))
    val = @inferred loss(nn, x_a, x_b)
    @test val isa T
    @test isapprox(val, expected; rtol = 10 * eps(T))

    # With β_align = 0 the loss is the cross term alone, which is not small and depends on x_b.
    loss_cross = CrossModalLoss(:a, :b; β_align = 0, β_cross = 1)
    cross = sum(abs2, Dec * z_a - vec(x_b))
    @test isapprox(loss_cross(nn, x_a, x_b), cross; rtol = 10 * eps(T))
    @test cross > 1e-2
    x_b2 = x_b .+ T(0.5)
    cross2 = sum(abs2, Dec * z_a - vec(x_b2))
    @test isapprox(loss_cross(nn, x_a, x_b2), cross2; rtol = 10 * eps(T))
    @test abs(loss_cross(nn, x_a, x_b2) - loss_cross(nn, x_a, x_b)) > T(1e-2) * cross

    # The cycle term ‖encode(b, decode(b, z_a)) - z_a‖ is zero for the weight-tied core with a
    # right-invertible frontend, and not zero for the untied core.
    z_a_nn = encode(nn, :a, x_a)
    cycle = norm(encode(nn, :b, decode(nn, :b, z_a_nn)) - z_a_nn)
    @test cycle <= 10 * eps(T) * max(one(T), norm(z_a_nn))
    z_a_sa = encode(nn_sa, :a, x_a)
    cycle_sa = norm(encode(nn_sa, :b, decode(nn_sa, :b, z_a_sa)) - z_a_sa)
    @test cycle_sa > T(1e-3) * norm(z_a_sa)

    # The self-reconstruction form of CharbonnierRelativeLoss has the latent term: on the untied
    # core, where that term is not zero, the value is the reconstruction term plus the latent
    # term, the value of the form with an external x̂ = decode(encode(x)).
    ε = T(1e-2)
    loss_c = CharbonnierRelativeLoss(; ε = ε, λ_lat = 1)
    x̂_a = decode(nn_sa, :a, encode(nn_sa, :a, x_a))
    rec = (sqrt(sum(abs2, x̂_a - x_a) + ε^2) - ε) / sqrt(sum(abs2, x_a) + ε^2)
    lat = sum(abs2, encode(nn_sa, :a, x̂_a) - z_a_sa) / (sum(abs2, x_a) + ε^2)
    @test lat > T(1e-3) * rec
    val_self = @inferred loss_c(nn_sa, :a, x_a)
    @test val_self isa T
    @test isapprox(val_self, rec + lat; rtol = 10 * eps(T))
    @test isapprox(loss_c(nn_sa, :a, x̂_a, x_a), rec + lat; rtol = 10 * eps(T))
end

@testset "Docstrings state the limits of the frontends and the losses" begin
    # The source text of every docstring of X, with its line breaks as spaces. `Docs.doc` has
    # no method until the REPL standard library loads, so the text comes from the module's
    # metadata.
    function docstring(X)
        binding = Base.Docs.Binding(GeometricMachineLearning, nameof(X))
        multidoc = Base.Docs.meta(GeometricMachineLearning)[binding]
        replace(join((join(d.text) for d in values(multidoc.docs)), "\n"), '\n' => ' ')
    end
    documented = (CanonicalSortLayer, CanonicalSlotLayer, InverseCanonicalSlotLayer,
        SymmetricSymplecticAutoencoder, BilinearStiefelLayer, LowRankStiefelLayer,
        MultiModalSymplecticAutoencoder, CharbonnierRelativeLoss, IsometricRolloutLoss,
        CrossModalLoss)
    absent = ("collective canonical slots", "relative modes canonically",
        "canonical quadratic energy", "β_cycle", "symplectic or Poisson frontend")
    for X in documented
        str = docstring(X)
        for phrase in absent
            @test !occursin(phrase, str)
        end
        @test !occursin(r"(Proposition|Theorem|Corollary|Section) \d", str)
    end
    for X in (CanonicalSortLayer, CanonicalSlotLayer, SymmetricSymplecticAutoencoder)
        @test occursin("keep their particle order", docstring(X))
    end
    @test occursin("fixed particle count", docstring(CanonicalSlotLayer))
    for X in (BilinearStiefelLayer, LowRankStiefelLayer, MultiModalSymplecticAutoencoder)
        @test occursin("declare on the parity coordinates", docstring(X))
    end
    @test occursin("requires a latent Hamiltonian", docstring(IsometricRolloutLoss))
    @test occursin("same physical state", docstring(CrossModalLoss))
    @test occursin("not symmetric", docstring(CrossModalLoss))
    @test !occursin("skips", docstring(CharbonnierRelativeLoss))
    @test occursin("have one element type", docstring(CharbonnierRelativeLoss))
end
