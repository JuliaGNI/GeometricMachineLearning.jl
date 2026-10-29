using GeometricMachineLearning
using GeometricMachineLearning: ForwardDiff, SymplecticCompression, params,
                                input_dimension, output_dimension
using GeometricMachineLearning.AbstractNeuralNetworks: GlorotUniform
using KernelAbstractions
using LinearAlgebra
using Test
import Random
import Zygote

include("../helpers/eltypes.jl")
include("../helpers/poisson.jl")

Random.seed!(42)

# Explicit permutation matrix on 2Md for a particle permutation σ ∈ S_M
function _particle_perm_matrix(σ::AbstractVector{<:Integer}, d::Int, ::Type{T}) where {T}
    M = length(σ)
    P_σ = zeros(T, M * d, M * d)
    for i in 1:M
        j = σ[i]
        P_σ[((i - 1) * d + 1):(i * d), ((j - 1) * d + 1):(j * d)] .= Matrix{T}(I, d, d)
    end
    [P_σ zeros(T, M * d, M * d); zeros(T, M * d, M * d) P_σ]
end

function _apply_particle_perm(
        qp::NamedTuple{(:q, :p)}, σ::AbstractVector{<:Integer})
    if ndims(qp.q) == 2
        (q = qp.q[:, σ], p = qp.p[:, σ])
    else
        (q = qp.q[:, σ, :], p = qp.p[:, σ, :])
    end
end

# All 24 permutations of S_4
const S4_PERMS = (
    [1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4], [1, 3, 4, 2], [1, 4, 2, 3], [1, 4, 3, 2],
    [2, 1, 3, 4], [2, 1, 4, 3], [2, 3, 1, 4], [2, 3, 4, 1], [2, 4, 1, 3], [2, 4, 3, 1],
    [3, 1, 2, 4], [3, 1, 4, 2], [3, 2, 1, 4], [3, 2, 4, 1], [3, 4, 1, 2], [3, 4, 2, 1],
    [4, 1, 2, 3], [4, 1, 3, 2], [4, 2, 1, 3], [4, 2, 3, 1], [4, 3, 1, 2], [4, 3, 2, 1]
)

# Biased activation with both even and odd parts to exercise oddization τ(x) = σ(x) - σ(-x)
act_biased(x) = tanh(x) + oftype(x, 0.35) * cos(x)

@testset "Clause 1: CanonicalSortLayer and InvertCanonicalSortLayer" begin
    # Constructor validation
    re_sort_m = r"^(?:ArgumentError: )?CanonicalSortLayer requires M >= 1, d >= 1 and 1 <= axis <= d, got M = 0, d = 2, axis = 1\.$"
    re_sort_ax = r"^(?:ArgumentError: )?CanonicalSortLayer requires M >= 1, d >= 1 and 1 <= axis <= d, got M = 4, d = 2, axis = 3\.$"
    re_sort_tp = r"^(?:ArgumentError: )?CanonicalSortLayer\{N, M, Axis\} requires M >= 1, Axis >= 1 and N = 2Md with d >= Axis, got N = 10, M = 4, Axis = 1\.$"
    re_inv_m = r"^(?:ArgumentError: )?InvertCanonicalSortLayer requires M >= 1 and d >= 1, got M = 0, d = 2\.$"
    re_inv_tp = r"^(?:ArgumentError: )?InvertCanonicalSortLayer\{N, M\} requires M >= 1 and N = 2Md with d >= 1, got N = 10, M = 4\.$"
    @test_throws re_sort_m CanonicalSortLayer(0, 2)
    @test_throws re_sort_ax CanonicalSortLayer(4, 2, 3)
    @test_throws re_sort_tp CanonicalSortLayer{10, 4, 1}()
    @test_throws re_inv_m InvertCanonicalSortLayer(0, 2)
    @test_throws re_inv_tp InvertCanonicalSortLayer{10, 4}()

    @test CanonicalSortLayer(4, 2) === CanonicalSortLayer{16, 4, 1}()
    @test CanonicalSortLayer(4, 2, 2) === CanonicalSortLayer{16, 4, 2}()
    @test InvertCanonicalSortLayer(4, 2) === InvertCanonicalSortLayer{16, 4}()
    @test inverse_layer(CanonicalSortLayer{16, 4, 1}()) ===
          InvertCanonicalSortLayer{16, 4}()
    @test inverse_layer(CanonicalSortLayer{16, 4, 2}()) ===
          InvertCanonicalSortLayer{16, 4}()
    @test inverse_layer(InvertCanonicalSortLayer{16, 4}()) ===
          CanonicalSortLayer{16, 4, 1}()
    @test inverse_layer(CanonicalSortLayer{16, 4, 1}) === InvertCanonicalSortLayer{16, 4}
    @test inverse_layer(InvertCanonicalSortLayer{16, 4}) === CanonicalSortLayer{16, 4, 1}
    @test inverse_layer(CanonicalSortLayer) === InvertCanonicalSortLayer
    @test inverse_layer(InvertCanonicalSortLayer) === CanonicalSortLayer
    @test parameterlength(CanonicalSortLayer(4, 2)) == 0
    @test parameterlength(InvertCanonicalSortLayer(4, 2)) == 0
    @test initialparameters(Random.default_rng(), GlorotUniform(),
        CanonicalSortLayer(4, 2), CPU(), Float64) === NamedTuple()
    @test initialparameters(Random.default_rng(), GlorotUniform(),
        InvertCanonicalSortLayer(4, 2), CPU(), Float64) === NamedTuple()

    # input_dimension and output_dimension are the length of the flattened (q, p) state
    for layer in (CanonicalSortLayer(4, 2), CanonicalSortLayer(4, 3, 2),
        InvertCanonicalSortLayer(4, 2), InvertCanonicalSortLayer(3, 1))
        z = randn(input_dimension(layer))
        @test input_dimension(layer) == output_dimension(layer)
        @test length(layer(z, NamedTuple())) == output_dimension(layer)
        @test_throws DimensionMismatch layer(randn(input_dimension(layer) + 2), NamedTuple())
    end

    # DimensionMismatch guards with anchored patterns
    l4_1 = CanonicalSortLayer(4, 2)
    inv4 = InvertCanonicalSortLayer(4, 2)
    ps_empty = NamedTuple()

    re_qp_size = r"^(?:DimensionMismatch: )?CanonicalSortLayer\{16, 4, 1\} expects q and p of matching size \(2, 4\) or \(2, 4, B\), got \(2, 3\) and \(2, 3\)\.$"
    re_qp_mismatch = r"^(?:DimensionMismatch: )?CanonicalSortLayer\{16, 4, 1\} expects q and p of matching size \(2, 4\) or \(2, 4, B\), got \(2, 4\) and \(2, 3\)\.$"
    re_qp_d = r"^(?:DimensionMismatch: )?CanonicalSortLayer\{16, 4, 1\} expects q and p of matching size \(2, 4\) or \(2, 4, B\), got \(3, 4\) and \(3, 4\)\.$"
    re_vec_dim = r"^(?:DimensionMismatch: )?CanonicalSortLayer\{16, 4, 1\} expects first dimension 16, got 10\.$"
    re_inv_size = r"^(?:DimensionMismatch: )?InvertCanonicalSortLayer\{16, 4\} expects q and p of matching size \(2, 4\) or \(2, 4, B\), got \(2, 3\) and \(2, 3\)\.$"
    re_inv_perm = r"^(?:DimensionMismatch: )?InvertCanonicalSortLayer\{16, 4\} expects perm of size \(4,\), got \(3,\)\.$"
    re_inv_perm_b = r"^(?:DimensionMismatch: )?InvertCanonicalSortLayer\{16, 4\} expects perm of size \(4, 2\), got \(4, 3\)\.$"
    re_inv_vec = r"^(?:DimensionMismatch: )?InvertCanonicalSortLayer\{16, 4\} expects first dimension 16, got 10\.$"

    @test_throws re_qp_size l4_1((q = zeros(2, 3), p = zeros(2, 3)), ps_empty)
    @test_throws re_qp_mismatch l4_1((q = zeros(2, 4), p = zeros(2, 3)), ps_empty)
    @test_throws re_qp_d l4_1((q = zeros(3, 4), p = zeros(3, 4)), ps_empty)
    @test_throws re_vec_dim l4_1(zeros(10), ps_empty)
    @test_throws re_inv_size inv4(
        (q = zeros(2, 3), p = zeros(2, 3), perm = [1, 2, 3]), ps_empty)
    @test_throws re_inv_perm inv4(
        (q = zeros(2, 4), p = zeros(2, 4), perm = [1, 2, 3]), ps_empty)
    @test_throws re_inv_perm_b inv4(
        (q = zeros(2, 4, 2), p = zeros(2, 4, 2), perm = zeros(Int, 4, 3)), ps_empty)
    @test_throws re_inv_vec inv4(zeros(10), ps_empty)

    @testset "Precision $T" for T in REAL_ELTYPES
        M = 4
        d = 2
        l = CanonicalSortLayer(M, d)
        inv_l = InvertCanonicalSortLayer(M, d)

        # Distinct particle positions placed so the sorting permutation is a 4-cycle
        # σ = [2, 3, 4, 1] with σ^2 = [3, 4, 1, 2] != id (catches M2!).
        # Particle 2 has smallest q[1, :], then 3, then 4, then 1:
        q_single = T[4.0 1.0 2.0 3.0;
                     -0.5 1.5 -2.0 0.7]
        # Distinct momenta that are NOT sorted in the same order as q (catches M1!):
        p_single = T[-3.0 2.5 -1.0 4.0;
                     1.2 -0.8 3.1 -2.2]
        qp_single = (q = q_single, p = p_single)

        out_single = @inferred l(qp_single, ps_empty)
        @test eltype(out_single.q) === T
        @test eltype(out_single.p) === T
        @test out_single.perm == [2, 3, 4, 1]
        # Verify σ is a 4-cycle (σ^2 != id)
        @test out_single.perm[out_single.perm] != [1, 2, 3, 4]
        @test issorted(@view(out_single.q[1, :]))
        # Identical permutation applied to both q and p
        @test out_single.q == q_single[:, out_single.perm]
        @test out_single.p == p_single[:, out_single.perm]

        # Exact zero-ULP round-trip
        rt_single = @inferred inv_l(out_single, ps_empty)
        @test eltype(rt_single.q) === T
        @test eltype(rt_single.p) === T
        @test rt_single == qp_single
        @test inv_l((q = out_single.q, p = out_single.p), ps_empty) ==
              (q = out_single.q, p = out_single.p)

        # Strict S_M-invariance across all 24 permutations π ∈ S_4
        for π_perm in S4_PERMS
            qp_π = _apply_particle_perm(qp_single, π_perm)
            out_π = l(qp_π, ps_empty)
            @test out_π.q == out_single.q
            @test out_π.p == out_single.p
            @test inv_l(out_π, ps_empty) == qp_π
        end

        # Exact canonical Jacobian D^T J_{2Md} D == J_{2Md} (catches M1!)
        J_2Md = poisson_matrix(2 * M * d, T)
        D_sort = ForwardDiff.jacobian(z -> l(z, ps_empty), vcat(vec(q_single), vec(p_single)))
        @test eltype(D_sort) === T
        @test transpose(D_sort) * J_2Md * D_sort == J_2Md

        # Axis = 2 sorting
        l_ax2 = CanonicalSortLayer(M, d, 2)
        out_ax2 = @inferred l_ax2(qp_single, ps_empty)
        @test issorted(@view(out_ax2.q[2, :]))
        @test inv_l(out_ax2, ps_empty) == qp_single

        # Collision ties: stable MergeSort preserves input order deterministically
        q_tied = T[2.0 1.0 1.0 2.0;
                   10.0 20.0 30.0 40.0]
        p_tied = T[1.0 2.0 3.0 4.0;
                   5.0 6.0 7.0 8.0]
        out_tied = l((q = q_tied, p = p_tied), ps_empty)
        @test out_tied.perm == [2, 3, 1, 4]
        @test out_tied.q[:, 1:2] == q_tied[:, 2:3]
        @test out_tied.q[:, 3:4] == q_tied[:, [1, 4]]
        @test inv_l(out_tied, ps_empty) == (q = q_tied, p = p_tied)

        # Batched 3D (d, M, B) with different non-involution permutations per batch sample
        B = 3
        q_batch = zeros(T, d, M, B)
        p_batch = zeros(T, d, M, B)
        q_batch[:, :, 1] .= q_single
        p_batch[:, :, 1] .= p_single
        q_batch[:, :, 2] .= q_single[:, [3, 1, 4, 2]]
        p_batch[:, :, 2] .= p_single[:, [3, 1, 4, 2]]
        q_batch[:, :, 3] .= q_single[:, [4, 3, 2, 1]]
        p_batch[:, :, 3] .= p_single[:, [4, 3, 2, 1]]
        qp_batch = (q = q_batch, p = p_batch)

        out_batch = @inferred l(qp_batch, ps_empty)
        @test eltype(out_batch.q) === T
        @test eltype(out_batch.p) === T
        @test size(out_batch.perm) == (M, B)
        for b in 1:B
            @test out_batch.q[:, :, b] == out_single.q
            @test out_batch.p[:, :, b] == out_single.p
        end
        rt_batch = @inferred inv_l(out_batch, ps_empty)
        @test rt_batch == qp_batch

        # Flattened vector (2Md) and snapshot matrix (2Md, B) forms
        z_vec = vcat(vec(q_single), vec(p_single))
        z_mat = vcat(reshape(q_batch, M * d, B), reshape(p_batch, M * d, B))
        z_vec_sorted = @inferred l(z_vec, ps_empty)
        z_mat_sorted = @inferred l(z_mat, ps_empty)
        @test eltype(z_vec_sorted) === T
        @test eltype(z_mat_sorted) === T
        @test z_vec_sorted == vcat(vec(out_single.q), vec(out_single.p))
        for b in 1:B
            @test z_mat_sorted[:, b] == z_vec_sorted
        end
        @test (@inferred inv_l(z_vec_sorted, ps_empty)) == z_vec_sorted
        @test (@inferred inv_l(z_mat_sorted, ps_empty)) == z_mat_sorted

        # Zygote differentiation through vector and batched matrix forward pass
        g_vec = Zygote.gradient(x -> sum(abs2, l(x, ps_empty)), z_vec)[1]
        @test g_vec ≈ 2 .* z_vec
        g_mat = Zygote.gradient(x -> sum(abs2, l(x, ps_empty)), z_mat)[1]
        @test g_mat ≈ 2 .* z_mat
    end
end

@testset "Clause 2: PairwiseSymplecticLayerQ and PairwiseSymplecticLayerP" begin
    # Constructor validation
    re_pw_m = r"^(?:ArgumentError: )?PairwiseSymplecticLayer requires M >= 1, d >= 1, and upscale >= 1, got M = 0, d = 2, upscale = 3\.$"
    re_pw_d = r"^(?:ArgumentError: )?PairwiseSymplecticLayer requires M >= 1, d >= 1, and upscale >= 1, got M = 4, d = 0, upscale = 3\.$"
    re_pw_u = r"^(?:ArgumentError: )?PairwiseSymplecticLayer requires M >= 1, d >= 1, and upscale >= 1, got M = 4, d = 2, upscale = 0\.$"
    @test_throws re_pw_m PairwiseSymplecticLayerQ(0, 2, 3, tanh)
    @test_throws re_pw_d PairwiseSymplecticLayerQ(4, 0, 3, tanh)
    @test_throws re_pw_u PairwiseSymplecticLayerP(4, 2, 0, tanh)

    lq_chk = PairwiseSymplecticLayerQ(4, 2, 3, tanh)
    lp_chk = PairwiseSymplecticLayerP(4, 2, 3, tanh)
    @test lq_chk isa PairwiseSymplecticLayerQ{16, 6, typeof(tanh)}
    @test lp_chk isa PairwiseSymplecticLayerP{16, 6, typeof(tanh)}
    @test inverse_layer(lq_chk) isa InversePairwiseSymplecticLayerQ{16, 6, typeof(tanh)}
    @test inverse_layer(lp_chk) isa InversePairwiseSymplecticLayerP{16, 6, typeof(tanh)}
    @test inverse_layer(inverse_layer(lq_chk)) === lq_chk
    @test inverse_layer(inverse_layer(lp_chk)) === lp_chk
    @test inverse_layer(PairwiseSymplecticLayerQ) === InversePairwiseSymplecticLayerQ
    @test inverse_layer(InversePairwiseSymplecticLayerQ) === PairwiseSymplecticLayerQ
    @test inverse_layer(PairwiseSymplecticLayerP) === InversePairwiseSymplecticLayerP
    @test inverse_layer(InversePairwiseSymplecticLayerP) === PairwiseSymplecticLayerP
    @test inverse_layer(PairwiseSymplecticLayerQ{16, 6, typeof(tanh)}) ===
          InversePairwiseSymplecticLayerQ{16, 6, typeof(tanh)}
    @test inverse_layer(InversePairwiseSymplecticLayerQ{16, 6, typeof(tanh)}) ===
          PairwiseSymplecticLayerQ{16, 6, typeof(tanh)}
    @test inverse_layer(PairwiseSymplecticLayerP{16, 6, typeof(tanh)}) ===
          InversePairwiseSymplecticLayerP{16, 6, typeof(tanh)}
    @test inverse_layer(InversePairwiseSymplecticLayerP{16, 6, typeof(tanh)}) ===
          PairwiseSymplecticLayerP{16, 6, typeof(tanh)}

    # DimensionMismatch guards with anchored patterns
    ps_chk = initialparameters(
        Random.default_rng(), GlorotUniform(), lq_chk, CPU(), Float64)
    @test parameterlength(lq_chk) == 6 * 2 + 6 + 6 * 2
    @test parameterlength(inverse_layer(lq_chk)) == parameterlength(lq_chk)
    # input_dimension and output_dimension are the length of the flattened (q, p) state
    for layer in (lq_chk, lp_chk, inverse_layer(lq_chk), inverse_layer(lp_chk))
        @test input_dimension(layer) == output_dimension(layer) == 16
        @test length(layer(randn(input_dimension(layer)), ps_chk)) ==
              output_dimension(layer)
    end
    re_pw_qp = r"^(?:DimensionMismatch: )?PairwiseSymplecticLayer expects q and p of matching size \(2, 4\), \(2, 4, B\), or \(8,\), got \(2, 3\) and \(2, 3\)\.$"
    re_pw_vec = r"^(?:DimensionMismatch: )?PairwiseSymplecticLayer expects first dimension 16, got 10\.$"
    @test_throws re_pw_qp lq_chk((q = zeros(2, 3), p = zeros(2, 3)), ps_chk)
    @test_throws re_pw_vec lq_chk(zeros(10), ps_chk)
    @test_throws re_pw_vec inverse_layer(lq_chk)(zeros(10), ps_chk)

    @testset "Precision $T" for T in REAL_ELTYPES
        M = 4
        d = 2
        upscale = 3

        for (layer, inv_layer, is_q) in (
            (PairwiseSymplecticLayerQ(M, d, upscale, act_biased),
            inverse_layer(PairwiseSymplecticLayerQ(M, d, upscale, act_biased)), true),
            (PairwiseSymplecticLayerP(M, d, upscale, act_biased),
            inverse_layer(PairwiseSymplecticLayerP(M, d, upscale, act_biased)), false)
        )
            ps = initialparameters(
                Random.default_rng(), GlorotUniform(), layer, CPU(), T;
                init_bias = GlorotUniform())
            @test eltype(ps.K_self) === T
            @test eltype(ps.b_self) === T
            @test eltype(ps.K_pair) === T

            q0 = randn(T, d, M)
            p0 = randn(T, d, M)
            qp0 = (q = q0, p = p0)
            z0 = vcat(vec(q0), vec(p0))

            # Closed-form oracle check against Done when 2 equation
            u0 = is_q ? q0 : p0
            K_s = ps.K_self
            b_s = ps.b_self
            K_p = ps.K_pair
            Δ_oracle = zeros(T, d, M)
            for i in 1:M
                u_i = u0[:, i]
                f_self = transpose(K_s) * act_biased.(K_s * u_i .+ b_s)
                f_pair = zeros(T, d)
                for j in 1:M
                    u_ij = K_p * (u_i .- u0[:, j])
                    τ_ij = act_biased.(u_ij) .- act_biased.(-u_ij)
                    f_pair .+= transpose(K_p) * τ_ij
                end
                Δ_oracle[:, i] .= f_self .+ f_pair ./ T(M)
            end

            out_qp = @inferred layer(qp0, ps)
            @test eltype(out_qp.q) === T && eltype(out_qp.p) === T
            if is_q
                @test out_qp.q == q0
                @test isapprox(out_qp.p, p0 .- Δ_oracle; atol = 20 * eps(T), rtol = 20 *
                                                                                    eps(T))
            else
                @test isapprox(out_qp.q, q0 .+ Δ_oracle; atol = 20 * eps(T), rtol = 20 *
                                                                                    eps(T))
                @test out_qp.p == p0
            end

            # Vector (2Md) and flattened (Md,) NamedTuple consistency
            out_vec = @inferred layer(z0, ps)
            @test eltype(out_vec) === T
            @test isapprox(out_vec, vcat(vec(out_qp.q), vec(out_qp.p));
                atol = 20 * eps(T), rtol = 20 * eps(T))
            out_flat_qp = @inferred layer((q = vec(q0), p = vec(p0)), ps)
            @test isapprox(out_flat_qp.q, vec(out_qp.q); atol = 20 * eps(T), rtol = 20 *
                                                                                    eps(T))
            @test isapprox(out_flat_qp.p, vec(out_qp.p); atol = 20 * eps(T), rtol = 20 *
                                                                                    eps(T))

            # (a) Strict S_M-equivariance under every π ∈ S_4
            for π_perm in S4_PERMS
                qp_π = _apply_particle_perm(qp0, π_perm)
                lhs = layer(qp_π, ps)
                rhs = _apply_particle_perm(out_qp, π_perm)
                err_eq = hypot(norm(lhs.q - rhs.q), norm(lhs.p - rhs.p))
                @test err_eq <= 20 * eps(T) * hypot(norm(rhs.q), norm(rhs.p))

                P_π = _particle_perm_matrix(π_perm, d, T)
                @test norm(layer(P_π * z0, ps) - P_π * out_vec) <=
                      20 * eps(T) * norm(out_vec)
            end

            # (b) Strictly canonical: ||D^T J_{2Md} D - J_{2Md}||_F <= 50 eps(T)
            J_2Md = poisson_matrix(2 * M * d, T)
            D = ForwardDiff.jacobian(z -> layer(z, ps), z0)
            @test eltype(D) === T
            @test norm(transpose(D) * J_2Md * D - J_2Md) <= 50 * eps(T)

            # (c) Exact analytical inverse via inverse_layer(l)(l(z, ps), ps) ≈ z to 20 eps(T)
            rt_qp = @inferred inv_layer(out_qp, ps)
            @test eltype(rt_qp.q) === T && eltype(rt_qp.p) === T
            @test hypot(norm(rt_qp.q - q0), norm(rt_qp.p - p0)) <=
                  20 * eps(T) * hypot(norm(q0), norm(p0))
            rt_vec = @inferred inv_layer(out_vec, ps)
            @test eltype(rt_vec) === T
            @test norm(rt_vec - z0) <= 20 * eps(T) * norm(z0)

            # (d) Total-momentum (or total-position) conservation when K_self = 0 (catches M3!)
            ps_zero_self = (
                K_self = zeros(T, size(ps.K_self)),
                b_self = ps.b_self,
                K_pair = ps.K_pair
            )
            out_zero_self = layer(qp0, ps_zero_self)
            if is_q
                p_sum_new = vec(sum(out_zero_self.p; dims = 2))
                p_sum_old = vec(sum(p0; dims = 2))
                @test norm(p_sum_new - p_sum_old) <=
                      20 * eps(T) * max(one(T), norm(p_sum_old))
            else
                q_sum_new = vec(sum(out_zero_self.q; dims = 2))
                q_sum_old = vec(sum(q0; dims = 2))
                @test norm(q_sum_new - q_sum_old) <=
                      20 * eps(T) * max(one(T), norm(q_sum_old))
            end

            # Batched 3D (d, M, B) and 2D (2Md, B) consistency and Zygote differentiation
            B = 3
            q_b = randn(T, d, M, B)
            p_b = randn(T, d, M, B)
            out_b = @inferred layer((q = q_b, p = p_b), ps)
            @test eltype(out_b.q) === T && eltype(out_b.p) === T
            for b in 1:B
                single_b = layer((q = q_b[:, :, b], p = p_b[:, :, b]), ps)
                @test isapprox(out_b.q[:, :, b], single_b.q; atol = 20 * eps(T), rtol = 20 *
                                                                                        eps(T))
                @test isapprox(out_b.p[:, :, b], single_b.p; atol = 20 * eps(T), rtol = 20 *
                                                                                        eps(T))
            end
            rt_b = @inferred inv_layer(out_b, ps)
            @test hypot(norm(rt_b.q - q_b), norm(rt_b.p - p_b)) <=
                  20 * eps(T) * hypot(norm(q_b), norm(p_b))

            z_mat = vcat(reshape(q_b, M * d, B), reshape(p_b, M * d, B))
            out_mat = @inferred layer(z_mat, ps)
            @test eltype(out_mat) === T
            @test isapprox(
                out_mat, vcat(reshape(out_b.q, M * d, B), reshape(out_b.p, M * d, B));
                atol = 20 * eps(T), rtol = 20 * eps(T))
            rt_mat = @inferred inv_layer(out_mat, ps)
            @test norm(rt_mat - z_mat) <= 20 * eps(T) * norm(z_mat)

            # Zygote gradients against forward-mode ForwardDiff gradients
            loss(x, p) = sum(abs2, layer(x, p))
            g_z, g_ps = Zygote.gradient(loss, z_mat, ps)
            @test eltype(g_z) === T
            @test eltype(g_ps.K_self) === T && eltype(g_ps.K_pair) === T
            @test isapprox(g_z, ForwardDiff.gradient(x -> loss(x, ps), z_mat);
                rtol = 50 * eps(T))
            for key in (:K_self, :b_self, :K_pair)
                g_fd = ForwardDiff.gradient(
                    w -> loss(z_mat, merge(ps, NamedTuple{(key,)}((w,)))), ps[key])
                @test isapprox(g_ps[key], g_fd; rtol = 50 * eps(T))
            end
        end
    end
end

@testset "Clause 3: CanonicalSlotLayer and InverseCanonicalSlotLayer" begin
    # Constructor edge guards from Decided at the edges
    re_slot_m = r"^(?:ArgumentError: )?CanonicalSlotLayer requires M >= 2, d >= 1 and 1 <= NSlots <= M, got M = 1, d = 2, NSlots = 1\.$"
    re_slot_d = r"^(?:ArgumentError: )?CanonicalSlotLayer requires M >= 2, d >= 1 and 1 <= NSlots <= M, got M = 4, d = 0, NSlots = 2\.$"
    re_slot_n0 = r"^(?:ArgumentError: )?CanonicalSlotLayer requires M >= 2, d >= 1 and 1 <= NSlots <= M, got M = 4, d = 2, NSlots = 0\.$"
    re_slot_nm = r"^(?:ArgumentError: )?CanonicalSlotLayer requires M >= 2, d >= 1 and 1 <= NSlots <= M, got M = 4, d = 2, NSlots = 5\.$"
    re_inv_slot = r"^(?:ArgumentError: )?InverseCanonicalSlotLayer requires M >= 2, d >= 1 and 1 <= NSlots <= M, got M = 1, d = 2, NSlots = 1\.$"
    re_slot_tp = r"^(?:ArgumentError: )?CanonicalSlotLayer\{N, NOut, M, NSlots\} requires M >= 2, 1 <= NSlots <= M, N = 2Md and NOut = 2 NSlots d with d >= 1, got N = 16, NOut = 6, M = 4, NSlots = 2\.$"
    @test_throws re_slot_m CanonicalSlotLayer(1, 2, 1)
    @test_throws re_slot_d CanonicalSlotLayer(4, 0, 2)
    @test_throws re_slot_n0 CanonicalSlotLayer(4, 2, 0)
    @test_throws re_slot_nm CanonicalSlotLayer(4, 2, 5)
    @test_throws re_inv_slot InverseCanonicalSlotLayer(1, 2, 1)
    @test_throws re_slot_tp CanonicalSlotLayer{16, 6, 4, 2}()

    l_slot = CanonicalSlotLayer(4, 2, 2)
    inv_slot = inverse_layer(l_slot)
    @test l_slot === CanonicalSlotLayer{16, 8, 4, 2}()
    @test inv_slot === InverseCanonicalSlotLayer{16, 8, 4, 2}()
    @test InverseCanonicalSlotLayer(4, 2, 2) === inv_slot
    @test inverse_layer(inv_slot) === l_slot
    @test inverse_layer(CanonicalSlotLayer{16, 8, 4, 2}) ===
          InverseCanonicalSlotLayer{16, 8, 4, 2}
    @test inverse_layer(InverseCanonicalSlotLayer{16, 8, 4, 2}) ===
          CanonicalSlotLayer{16, 8, 4, 2}
    @test inverse_layer(CanonicalSlotLayer) === InverseCanonicalSlotLayer
    @test inverse_layer(InverseCanonicalSlotLayer) === CanonicalSlotLayer

    # input_dimension and output_dimension are the lengths of the flattened particle and
    # slot states
    for (M, d, NSlots) in ((4, 2, 2), (5, 3, 3), (3, 1, 1))
        enc_l = CanonicalSlotLayer(M, d, NSlots)
        dec_l = inverse_layer(enc_l)
        ps_l = initialparameters(
            Random.default_rng(), GlorotUniform(), enc_l, CPU(), Float64)
        @test input_dimension(enc_l) == output_dimension(dec_l) == 2 * M * d
        @test output_dimension(enc_l) == input_dimension(dec_l) == 2 * NSlots * d
        @test length(enc_l(randn(input_dimension(enc_l)), ps_l)) == output_dimension(enc_l)
        @test length(dec_l(randn(input_dimension(dec_l)), ps_l)) == output_dimension(dec_l)
    end

    # DimensionMismatch guards with anchored patterns
    ps_slot_chk = initialparameters(
        Random.default_rng(), GlorotUniform(), l_slot, CPU(), Float64)
    re_slot_qp = r"^(?:DimensionMismatch: )?CanonicalSlotLayer\{16, 8, 4, 2\} expects q and p of matching size \(2, 4\) or \(2, 4, B\), got \(2, 3\) and \(2, 3\)\.$"
    re_slot_vec = r"^(?:DimensionMismatch: )?CanonicalSlotLayer\{16, 8, 4, 2\} expects first dimension 16, got 10\.$"
    re_inv_slot_qp = r"^(?:DimensionMismatch: )?InverseCanonicalSlotLayer\{16, 8, 4, 2\} expects q and p of matching size \(2, 2\) or \(2, 2, B\), got \(2, 3\) and \(2, 3\)\.$"
    re_inv_slot_vec = r"^(?:DimensionMismatch: )?InverseCanonicalSlotLayer\{16, 8, 4, 2\} expects first dimension 8, got 10\.$"
    @test_throws re_slot_qp l_slot((q = zeros(2, 3), p = zeros(2, 3)), ps_slot_chk)
    @test_throws re_slot_vec l_slot(zeros(10), ps_slot_chk)
    @test_throws re_inv_slot_qp inv_slot((q = zeros(2, 3), p = zeros(2, 3)), ps_slot_chk)
    @test_throws re_inv_slot_vec inv_slot(zeros(10), ps_slot_chk)

    # The slots are invariant under a relabelling of the particles
    for (M, NSlots) in ((2, 2), (3, 3), (4, 3), (5, 2))
        d = 2
        l_inv = CanonicalSlotLayer(M, d, NSlots)
        ps_inv = initialparameters(
            Random.default_rng(), GlorotUniform(), l_inv, CPU(), Float64)
        q_r = randn(d, M)
        p_r = randn(d, M)
        out_r = l_inv((q = q_r, p = p_r), ps_inv)
        for π_r in [[collect(M:-1:1)]; [Random.randperm(M) for _ in 1:10]]
            out_π = l_inv((q = q_r[:, π_r], p = p_r[:, π_r]), ps_inv)
            @test out_π.q == out_r.q && out_π.p == out_r.p
        end
    end

    @testset "Precision $T" for T in REAL_ELTYPES
        M = 5
        NSlots = 3
        d = 2
        layer = CanonicalSlotLayer(M, d, NSlots)
        dec_layer = inverse_layer(layer)

        ps = initialparameters(Random.default_rng(), GlorotUniform(), layer, CPU(), T)
        @test ps.Φ_rel isa StiefelManifold{T}
        @test size(ps.Φ_rel) == (M - 1, NSlots - 1)
        @test parameterlength(layer) == (M - 1) * (NSlots - 1) - ((NSlots - 1) * NSlots) ÷ 2
        @test parameterlength(dec_layer) == parameterlength(layer)

        # Single-snapshot (d, M) with unordered relative mode squared norms (catches M5!)
        # Construct relative modes directly via H_rel so that the particles are already sorted
        # along row 1 of q and the M - 1 = 4 modes have unordered squared norms:
        H_rel_ref = zeros(T, M, M - 1)
        for k in 1:(M - 1)
            s = inv(sqrt(T(k * (k + 1))))
            H_rel_ref[1:k, k] .= s
            H_rel_ref[k + 1, k] = -T(k) * s
        end
        q_cm_ref = reshape(T[1.2, -0.7], d, 1)
        p_cm_ref = reshape(T[-0.4, 0.9], d, 1)
        # Row 1 of q_rel is H_rel^T v for the ascending zero-sum offsets v = [-2, -1, 0, 1, 2],
        # so the particles are in the sorted chamber; its squared entries are [0.5, 1.5, 3, 5].
        # Mode squared norms:
        # k = 1: ||q||^2 = 0.5, ||p||^2 = 1.0  -> total = 1.5
        # k = 2: ||q||^2 = 1.5, ||p||^2 = 16.0 -> total = 17.5
        # k = 3: ||q||^2 = 3.0, ||p||^2 = 0.0  -> total = 3.0
        # k = 4: ||q||^2 = 5.0, ||p||^2 = 9.0  -> total = 14.0
        # Here:
        # - total norm order: σ_rel = [2, 4, 3, 1] (17.5 > 14.0 > 3.0 > 1.5)
        # - q_rel norm order: [4, 3, 2, 1] (5.0 > 3.0 > 1.5 > 0.5)
        # - p_rel norm order: [2, 4, 1, 3] (16.0 > 9.0 > 1.0 > 0.0)
        # - unsorted order:   [1, 2, 3, 4]
        v_sorted = T[-2, -1, 0, 1, 2]
        q_rel_in = vcat(transpose(transpose(H_rel_ref) * v_sorted), zeros(T, 1, M - 1))
        p_rel_in = T[1.0 0.0 0.0 3.0;
                     0.0 4.0 0.0 0.0]
        ones_cm = fill(inv(sqrt(T(M))), 1, M)
        q0 = q_cm_ref * ones_cm .+ q_rel_in * transpose(H_rel_ref)
        p0 = p_cm_ref * ones_cm .+ p_rel_in * transpose(H_rel_ref)
        @test issorted(q0[1, :])
        qp0 = (q = q0, p = p0)

        slots_qp = @inferred layer(qp0, ps)
        @test eltype(slots_qp.q) === T && eltype(slots_qp.p) === T
        @test size(slots_qp.q) == (d, NSlots)
        @test size(slots_qp.p) == (d, NSlots)

        # Slot 1 is intact center-of-mass (Q_cm, P_cm) = (1 / sqrt(M)) ∑_{i=1}^M (q_i, p_i)
        Q_cm_expected = vec(sum(q0; dims = 2)) ./ sqrt(T(M))
        P_cm_expected = vec(sum(p0; dims = 2)) ./ sqrt(T(M))
        @test isapprox(slots_qp.q[:, 1], Q_cm_expected; atol = 20 * eps(T), rtol = 20 *
                                                                                   eps(T))
        @test isapprox(slots_qp.p[:, 1], P_cm_expected; atol = 20 * eps(T), rtol = 20 *
                                                                                   eps(T))

        # Relative slots 2:NSlots match the sorted oracle (q_rel[:, σ_rel] * Φ_rel, p_rel[:, σ_rel] * Φ_rel) (catches M5!)
        σ_rel_oracle = [2, 4, 3, 1]
        q_slot_rel_oracle = q_rel_in[:, σ_rel_oracle] * Matrix(ps.Φ_rel)
        p_slot_rel_oracle = p_rel_in[:, σ_rel_oracle] * Matrix(ps.Φ_rel)
        @test isapprox(slots_qp.q[:, 2:NSlots], q_slot_rel_oracle;
            atol = 20 * eps(T), rtol = 20 * eps(T))
        @test isapprox(slots_qp.p[:, 2:NSlots], p_slot_rel_oracle;
            atol = 20 * eps(T), rtol = 20 * eps(T))

        # Canonical Poisson-map identity D J_{2Md} D^T == J_{2 NSlots d} for the encoder Jacobian D (catches M5!)
        J_2Md = poisson_matrix(2 * M * d, T)
        J_2NSlots_d = poisson_matrix(2 * NSlots * d, T)
        z0_unordered = vcat(vec(q0), vec(p0))
        D_enc = ForwardDiff.jacobian(z -> layer(z, ps), z0_unordered)
        @test eltype(D_enc) === T
        @test norm(D_enc * J_2Md * transpose(D_enc) - J_2NSlots_d) <= 50 * eps(T)

        # Decoder lift: preserves (Q_cm, P_cm) to 20 eps(T) and satisfies
        # (∇Ψ_slot^dec)^T J_{2Md} ∇Ψ_slot^dec == J_{2 NSlots d} to 50 eps(T) (catches M4!)
        q_slot_in = randn(T, d, NSlots)
        p_slot_in = randn(T, d, NSlots)
        qp_lifted = @inferred dec_layer((q = q_slot_in, p = p_slot_in), ps)
        @test eltype(qp_lifted.q) === T && eltype(qp_lifted.p) === T
        @test size(qp_lifted.q) == (d, M) && size(qp_lifted.p) == (d, M)

        Q_cm_lifted = vec(sum(qp_lifted.q; dims = 2)) ./ sqrt(T(M))
        P_cm_lifted = vec(sum(qp_lifted.p; dims = 2)) ./ sqrt(T(M))
        @test norm(Q_cm_lifted - q_slot_in[:, 1]) <=
              20 * eps(T) * max(one(T), norm(q_slot_in[:, 1]))
        @test norm(P_cm_lifted - p_slot_in[:, 1]) <=
              20 * eps(T) * max(one(T), norm(p_slot_in[:, 1]))

        z_slot_in = vcat(vec(q_slot_in), vec(p_slot_in))
        z_lifted = @inferred dec_layer(z_slot_in, ps)
        @test eltype(z_lifted) === T
        @test isapprox(z_lifted, vcat(vec(qp_lifted.q), vec(qp_lifted.p));
            atol = 20 * eps(T), rtol = 20 * eps(T))

        D_dec = ForwardDiff.jacobian(z -> dec_layer(z, ps), z_slot_in)
        @test eltype(D_dec) === T
        @test norm(transpose(D_dec) * J_2Md * D_dec - J_2NSlots_d) <= 50 * eps(T)

        # Exact right-inversion on the canonical chamber where Φ_rel embeds into descending relative modes:
        # Use Φ_rel = I_{M-1, NSlots-1} and slot inputs with descending squared norms so the lifted
        # relative modes are already in the canonical chamber (σ_rel = id). The relative slots have
        # a zero first q row, so the lifted particles tie on row 1 of q and the stable particle
        # sort keeps their order.
        Φ_id = zeros(T, M - 1, NSlots - 1)
        for k in 1:(NSlots - 1)
            Φ_id[k, k] = one(T)
        end
        ps_chamber = (Φ_rel = StiefelManifold(Φ_id),)
        q_slot_ch = T[0.5 0.0 0.0;
                      -0.3 3.0 1.0]
        p_slot_ch = T[0.2 -1.5 0.4;
                      0.7 1.0 -0.2]
        z_slot_ch = vcat(vec(q_slot_ch), vec(p_slot_ch))
        z_lifted_ch = @inferred dec_layer(z_slot_ch, ps_chamber)
        z_slot_rt = @inferred layer(z_lifted_ch, ps_chamber)
        @test eltype(z_slot_rt) === T
        @test norm(z_slot_rt - z_slot_ch) <= 50 * eps(T) * norm(z_slot_ch)
        qp_lifted_ch = @inferred dec_layer((q = q_slot_ch, p = p_slot_ch), ps_chamber)
        qp_slot_rt = @inferred layer(qp_lifted_ch, ps_chamber)
        @test hypot(norm(qp_slot_rt.q - q_slot_ch), norm(qp_slot_rt.p - p_slot_ch)) <=
              50 * eps(T) * hypot(norm(q_slot_ch), norm(p_slot_ch))

        # Degenerate NSlots == 1 edge case: only center-of-mass slot kept, no Φ_rel allocated
        l_cm = CanonicalSlotLayer(M, d, 1)
        inv_cm = inverse_layer(l_cm)
        ps_cm = initialparameters(Random.default_rng(), GlorotUniform(), l_cm, CPU(), T)
        @test ps_cm === NamedTuple()
        @test parameterlength(l_cm) == 0
        out_cm = @inferred l_cm(qp0, ps_cm)
        @test size(out_cm.q) == (d, 1) && size(out_cm.p) == (d, 1)
        @test isapprox(out_cm.q[:, 1], Q_cm_expected; atol = 20 * eps(T), rtol = 20 *
                                                                                 eps(T))
        @test isapprox(out_cm.p[:, 1], P_cm_expected; atol = 20 * eps(T), rtol = 20 *
                                                                                 eps(T))
        lift_cm = @inferred inv_cm(out_cm, ps_cm)
        @test size(lift_cm.q) == (d, M) && size(lift_cm.p) == (d, M)
        @test isapprox(l_cm(lift_cm, ps_cm).q, out_cm.q; atol = 20 * eps(T), rtol = 20 *
                                                                                    eps(T))
        @test isapprox(l_cm(lift_cm, ps_cm).p, out_cm.p; atol = 20 * eps(T), rtol = 20 *
                                                                                    eps(T))
        D_cm = ForwardDiff.jacobian(z -> inv_cm(z, ps_cm), vcat(vec(out_cm.q), vec(out_cm.p)))
        J_2d = poisson_matrix(2 * d, T)
        @test norm(transpose(D_cm) * J_2Md * D_cm - J_2d) <= 50 * eps(T)

        # Batched (d, M, B) and (2Md, B) consistency with single-sample evaluation
        B = 3
        q_b = randn(T, d, M, B)
        p_b = randn(T, d, M, B)
        q_b[:, :, 1] .= q0
        p_b[:, :, 1] .= p0
        out_b = @inferred layer((q = q_b, p = p_b), ps)
        @test size(out_b.q) == (d, NSlots, B) && size(out_b.p) == (d, NSlots, B)
        for b in 1:B
            s_b = layer((q = q_b[:, :, b], p = p_b[:, :, b]), ps)
            @test isapprox(out_b.q[:, :, b], s_b.q; atol = 20 * eps(T), rtol = 20 * eps(T))
            @test isapprox(out_b.p[:, :, b], s_b.p; atol = 20 * eps(T), rtol = 20 * eps(T))
        end
        lift_b = @inferred dec_layer(
            (q = repeat(q_slot_ch, 1, 1, B), p = repeat(p_slot_ch, 1, 1, B)), ps_chamber)
        @test size(lift_b.q) == (d, M, B) && size(lift_b.p) == (d, M, B)
        rt_b = @inferred layer(lift_b, ps_chamber)
        for b in 1:B
            @test hypot(norm(rt_b.q[:, :, b] - q_slot_ch), norm(rt_b.p[:, :, b] -
                                                                p_slot_ch)) <=
                  50 * eps(T) * hypot(norm(q_slot_ch), norm(p_slot_ch))
        end
    end
end

@testset "Clause 4: SymmetricSymplecticAutoencoder" begin
    @test SymmetricSymplecticAutoencoder <: SymplecticCompression

    # Constructor validation with anchored patterns
    re_ssae_m = r"^(?:ArgumentError: )?SymmetricSymplecticAutoencoder requires M >= 1 and d >= 1, got M = 0, d = 2\.$"
    re_ssae_fe = r"^(?:ArgumentError: )?frontend must be :sort, :pairwise, or :slot, got :unknown\.$"
    re_ssae_ax = r"^(?:ArgumentError: )?sort_axis \(3\) must satisfy 1 <= sort_axis <= d \(2\)\.$"
    re_ssae_pw = r"^(?:ArgumentError: )?n_pairwise_layers \(-1\) must be non-negative and pairwise_upscale \(3\) must be at least 1\.$"
    re_ssae_sl = r"^(?:ArgumentError: )?frontend = :slot requires M >= 2 and 1 <= n_slots <= M, got M = 4, n_slots = 5\.$"
    re_ssae_chain = r"^(?:ArgumentError: )?A SymmetricSymplecticAutoencoder has no Chain of untied layers: build it with NeuralNetwork\(arch\) and take encoder\(nn\) and decoder\(nn\)\.$"
    @test_throws re_ssae_m SymmetricSymplecticAutoencoder(0, 2, 2)
    @test_throws re_ssae_fe SymmetricSymplecticAutoencoder(4, 2, 2; frontend = :unknown)
    @test_throws re_ssae_ax SymmetricSymplecticAutoencoder(4, 2, 2; sort_axis = 3)
    @test_throws re_ssae_pw SymmetricSymplecticAutoencoder(4, 2, 2; n_pairwise_layers = -1)
    @test_throws re_ssae_sl SymmetricSymplecticAutoencoder(
        4, 2, 2; frontend = :slot, n_slots = 5)
    @test_throws re_ssae_chain Chain(SymmetricSymplecticAutoencoder(4, 2, 2))

    # Boundary constructor values (M = 1, d = 1; sort_axis = d; n_slots = 1 and n_slots = M)
    @test SymmetricSymplecticAutoencoder(1, 1, 2; frontend = :sort, sort_axis = 1).n_particles ==
          1
    @test SymmetricSymplecticAutoencoder(4, 2, 2; frontend = :sort, sort_axis = 2).sort_axis ==
          2
    @test SymmetricSymplecticAutoencoder(4, 2, 2; frontend = :slot, n_slots = 1).n_slots ==
          1
    @test SymmetricSymplecticAutoencoder(4, 2, 2; frontend = :slot, n_slots = 4).n_slots ==
          4

    # Helper shape guards
    re_flat_qp = r"^(?:DimensionMismatch: )?Expected particle \(q, p\) of size \(2, 4\), \(2, 4, B\), or \(8,\), got \(2, 3\) and \(2, 3\)\.$"
    re_unflat_qp = r"^(?:DimensionMismatch: )?Expected flattened particle \(q, p\) of first dimension 8, got \(6,\) and \(6,\)\.$"
    @test_throws re_flat_qp GeometricMachineLearning._flatten_particle_qp(
        (q = zeros(2, 3), p = zeros(2, 3)), 4, 2)
    @test_throws re_unflat_qp GeometricMachineLearning._unflatten_particle_qp(
        (q = zeros(6), p = zeros(6)), 4, 2)

    @testset "Frontend $fe ($T)" for fe in (:sort, :pairwise, :slot), T in REAL_ELTYPES

        M = 4
        d = 2
        reduced_dim = 2
        # For :sort and :slot, standalone right-inversion encoder(nn)(decoder(nn)(ξ)) ≈ ξ holds for ξ
        # whose decoded state lies in the canonical chamber, where the sorting permutations on the decoded
        # state are the identity (and globally on R^{2n} for :pairwise; for :sort and :slot,
        # encoder(nn)(D_π * decoder(nn)(ξ)) ≈ ξ also holds for every π ∈ S_M).
        # With n_layers = 0, n_ambient_layers = 0 and Stiefel blocks set to identity embeddings, we can
        # also test on an explicit canonical-chamber latent state, as well as with general random weights
        # on a chamber-aligned latent state!
        arch = SymmetricSymplecticAutoencoder(
            M, d, reduced_dim;
            frontend = fe,
            n_slots = 2,
            n_pairwise_layers = 2,
            pairwise_upscale = 3,
            n_blocks = 1,
            n_layers = 1,
            n_ambient_layers = 1,
            sympnet_upscale = 2
        )
        nn = NeuralNetwork(arch, CPU(), T)
        if fe === :slot
            # With n_layers = 0 and n_ambient_layers = 0, the core Stiefel leaf Φ ∈ St(1, n_slots d)
            # puts ξ_q into row 1 of q in slot 2 (index d + 1), and ξ_p into the same entry of p.
            # The slot Stiefel leaf Φ_rel = H_rel^T w / ||H_rel^T w|| for the ascending zero-sum
            # w = [-3, 0.9, 1, 1.1] lifts slot 2 to particle offsets proportional to w, so for
            # ξ_q > 0 the decoded particles are sorted along row 1 of q (σ = id), and the entries
            # of Φ_rel have descending magnitudes, so the relative modes are in the canonical
            # chamber σ_rel = id!
            arch_ch = SymmetricSymplecticAutoencoder(
                M, d, reduced_dim;
                frontend = :slot,
                n_slots = 2,
                n_blocks = 1,
                n_layers = 0,
                n_ambient_layers = 0,
                sympnet_upscale = 2
            )
            nn = NeuralNetwork(arch_ch, CPU(), T)
            Φ_core = zeros(T, 2 * d, reduced_dim ÷ 2)
            Φ_core[d + 1, 1] = one(T)
            copyto!(parent(first(values(params(nn))).weight), Φ_core)
            H_rel_4 = zeros(T, M, M - 1)
            for k in 1:(M - 1)
                s = inv(sqrt(T(k * (k + 1))))
                H_rel_4[1:k, k] .= s
                H_rel_4[k + 1, k] = -T(k) * s
            end
            Φ_rel_mat = reshape(transpose(H_rel_4) * T[-3.0, 0.9, 1.0, 1.1], M - 1, 1)
            Φ_rel_mat ./= norm(Φ_rel_mat)
            @test issorted(abs.(vec(Φ_rel_mat)); rev = true)
            copyto!(parent(last(values(params(nn))).Φ_rel), Φ_rel_mat)
        elseif fe === :sort
            # Set the core Stiefel leaf Φ ∈ St(1, M*d) to have strictly increasing particle-axis-1 components
            # and zero out the small SympNet shear biases/weights so decoder(nn)(ξ) for ξ_1 > 0 lands inside
            # the canonical Weyl chamber q_1 < q_2 < ... < q_M (σ = id), while still exercising the full
            # CanonicalSortLayer / InvertCanonicalSortLayer pipeline and S_M permutation invariance!
            arch_ch = SymmetricSymplecticAutoencoder(
                M, d, reduced_dim;
                frontend = :sort,
                n_blocks = 1,
                n_layers = 0,
                n_ambient_layers = 0,
                sympnet_upscale = 2
            )
            nn = NeuralNetwork(arch_ch, CPU(), T)
            Φ_core = zeros(T, M * d, reduced_dim ÷ 2)
            # Particle k has axis-1 index (k - 1) * d + 1; assign strictly increasing weights:
            weights_q1 = T[-3.0, -1.0, 1.0, 3.0]
            weights_q1 ./= norm(weights_q1)
            for k in 1:M
                Φ_core[(k - 1) * d + 1, 1] = weights_q1[k]
            end
            copyto!(parent(first(values(params(nn))).weight), Φ_core)
        end
        enc = encoder(nn)
        dec = decoder(nn)
        @test parameterlength(nn) == parameterlength(dec)
        @test parameterlength(enc) == parameterlength(dec)

        # Single-snapshot and batched (q, p) inputs + vector/matrix inputs
        q_s = randn(T, d, M)
        p_s = randn(T, d, M)
        qp_s = (q = q_s, p = p_s)
        B = 3
        q_b = randn(T, d, M, B)
        p_b = randn(T, d, M, B)
        qp_b = (q = q_b, p = p_b)

        # @inferred on single-snapshot and batched (q, p) inputs
        ξ_s = @inferred enc(qp_s)
        ξ_b = @inferred enc(qp_b)
        @test eltype(ξ_s.q) === T && eltype(ξ_s.p) === T
        @test eltype(ξ_b.q) === T && eltype(ξ_b.p) === T
        @test length(ξ_s.q) == reduced_dim ÷ 2 && length(ξ_s.p) == reduced_dim ÷ 2
        @test size(ξ_b.q) == (reduced_dim ÷ 2, B) && size(ξ_b.p) == (reduced_dim ÷ 2, B)

        # The latent of :sort and :slot is invariant under a relabelling of the particles
        if fe !== :pairwise
            for π_perm in S4_PERMS
                ξ_π = enc(_apply_particle_perm(qp_s, π_perm))
                @test ξ_π.q == ξ_s.q && ξ_π.p == ξ_s.p
            end
        end

        rec_qp_s = @inferred nn(qp_s)
        rec_qp_b = @inferred nn(qp_b)
        @test eltype(rec_qp_s.q) === T && eltype(rec_qp_s.p) === T
        @test eltype(rec_qp_b.q) === T && eltype(rec_qp_b.p) === T
        @test size(rec_qp_s.q) == (d, M) && size(rec_qp_s.p) == (d, M)
        @test size(rec_qp_b.q) == (d, M, B) && size(rec_qp_b.p) == (d, M, B)

        # Also test vector and matrix forms of nn(z)
        z_s_in = vcat(vec(q_s), vec(p_s))
        z_b_in = vcat(reshape(q_b, M * d, B), reshape(p_b, M * d, B))
        rec_z_s = @inferred nn(z_s_in)
        rec_z_b = @inferred nn(z_b_in)
        @test eltype(rec_z_s) === T && eltype(rec_z_b) === T
        @test isapprox(rec_z_s, vcat(vec(rec_qp_s.q), vec(rec_qp_s.p));
            atol = 20 * eps(T), rtol = 20 * eps(T))
        @test isapprox(
            rec_z_b, vcat(reshape(rec_qp_b.q, M * d, B), reshape(rec_qp_b.p, M * d, B));
            atol = 20 * eps(T), rtol = 20 * eps(T))

        # Exact latent right-inversion encoder(nn)(decoder(nn)(ξ)) ≈ ξ to 50 eps(T)
        # (with ξ_q > 0 so decoded state for :sort has q_1 < q_2 < ... < q_M, i.e., σ = id)
        ξ_vec = T[1.25, -0.75]
        ξ_mat = T[1.25 0.85 1.65;
                  -0.75 0.45 -1.15]
        ξ_nt_s = (
            q = ξ_vec[1:(reduced_dim ÷ 2)], p = ξ_vec[(reduced_dim ÷ 2 + 1):reduced_dim])
        ξ_nt_b = (q = ξ_mat[1:(reduced_dim ÷ 2), :],
            p = ξ_mat[(reduced_dim ÷ 2 + 1):reduced_dim, :])

        z_dec_vec = @inferred dec(ξ_vec)
        z_dec_mat = @inferred dec(ξ_mat)
        @test eltype(z_dec_vec) === T && eltype(z_dec_mat) === T
        ξ_rt_vec = @inferred enc(z_dec_vec)
        ξ_rt_mat = @inferred enc(z_dec_mat)
        @test eltype(ξ_rt_vec) === T && eltype(ξ_rt_mat) === T
        @test norm(ξ_rt_vec - ξ_vec) <= 50 * eps(T) * norm(ξ_vec)
        @test norm(ξ_rt_mat - ξ_mat) <= 50 * eps(T) * norm(ξ_mat)

        # For frontend = :sort and :slot, also verify encoder(nn)(D_π * decoder(nn)(ξ)) ≈ ξ for
        # every π ∈ S_4, and for :sort the perm-threaded reconstruction
        # nn(D_π * decoder(nn)(ξ)) ≈ D_π * decoder(nn)(ξ)!
        if fe !== :pairwise
            for π_perm in S4_PERMS
                P_π = _particle_perm_matrix(π_perm, d, T)
                z_π = P_π * z_dec_vec
                @test norm(enc(z_π) - ξ_vec) <= 50 * eps(T) * norm(ξ_vec)
                fe === :sort && @test norm(nn(z_π) - z_π) <= 50 * eps(T) * norm(z_π)
            end
        end

        qp_dec_s = @inferred dec(ξ_nt_s)
        qp_dec_b = @inferred dec(ξ_nt_b)
        @test eltype(qp_dec_s.q) === T && eltype(qp_dec_s.p) === T
        @test eltype(qp_dec_b.q) === T && eltype(qp_dec_b.p) === T
        @test size(qp_dec_s.q) == (d, M) && size(qp_dec_s.p) == (d, M)
        @test size(qp_dec_b.q) == (d, M, B) && size(qp_dec_b.p) == (d, M, B)

        ξ_rt_nt_s = @inferred enc(qp_dec_s)
        ξ_rt_nt_b = @inferred enc(qp_dec_b)
        @test hypot(norm(ξ_rt_nt_s.q - ξ_nt_s.q), norm(ξ_rt_nt_s.p - ξ_nt_s.p)) <=
              50 * eps(T) * hypot(norm(ξ_nt_s.q), norm(ξ_nt_s.p))
        @test hypot(norm(ξ_rt_nt_b.q - ξ_nt_b.q), norm(ξ_rt_nt_b.p - ξ_nt_b.p)) <=
              50 * eps(T) * hypot(norm(ξ_nt_b.q), norm(ξ_nt_b.p))

        # Decoder symplecticity (∇dec)^T J_{2Md} (∇dec) == J_{reduced_dim}
        J_2Md = poisson_matrix(2 * M * d, T)
        J_red = poisson_matrix(reduced_dim, T)
        D_dec = ForwardDiff.jacobian(x -> dec(x), ξ_vec)
        @test norm(transpose(D_dec) * J_2Md * D_dec - J_red) <= 100 * eps(T)
    end
end
