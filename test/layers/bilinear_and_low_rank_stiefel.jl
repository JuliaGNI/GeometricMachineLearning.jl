using GeometricMachineLearning
using GeometricMachineLearning: StiefelManifold
using GeometricOptimizers: solve!
using AbstractNeuralNetworks: AbstractExplicitLayer, Chain, NeuralNetwork, CPU,
                              GlorotUniform, initialparameters, parameterlength
using LinearAlgebra: I, qr, svd, norm, tr, Diagonal, diag
using Random: MersenneTwister
using Test
import Zygote

const REAL_ELTYPES = (Float32, Float64)

function _random_stiefel(rng, ::Type{T}, N::Int, m::Int) where {T <: Real}
    A = randn(rng, T, N, m)
    return Matrix{T}(qr(A).Q)[:, 1:m]
end

function _canonical_symplectic_matrix(::Type{T}, two_n::Int) where {T <: Real}
    n = two_n ÷ 2
    Z = zeros(T, n, n)
    Id = Matrix{T}(I, n, n)
    return [Z Id; -Id Z]
end

@testset "BilinearStiefelLayer constructor, guards, parameters" begin
    for T in REAL_ELTYPES
        rng = MersenneTwister(42)
        Nx, Nv, mx, mv = 6, 8, 3, 2
        l = BilinearStiefelLayer(Nx, Nv, mx, mv)
        inv_l = @inferred inverse_layer(l)
        @test inv_l isa TransposeBilinearStiefelLayer
        @test (@inferred inverse_layer(inv_l)) === l

        # Constructor guards
        @test_throws r"^(?:ArgumentError: )?Nx must be ≥ 1; got Nx = 0$" BilinearStiefelLayer(0, 8, 1, 1)
        @test_throws r"^(?:ArgumentError: )?Nv must be an even integer ≥ 2; got Nv = 7$" BilinearStiefelLayer(
            6, 7, 2, 2)
        @test_throws r"^(?:ArgumentError: )?Nv must be an even integer ≥ 2; got Nv = 0$" BilinearStiefelLayer(
            6, 0, 2, 1)
        @test_throws r"^(?:ArgumentError: )?mx must satisfy 1 ≤ mx ≤ Nx; got mx = 0, Nx = 6$" BilinearStiefelLayer(
            6, 8, 0, 2)
        @test_throws r"^(?:ArgumentError: )?mx must satisfy 1 ≤ mx ≤ Nx; got mx = 7, Nx = 6$" BilinearStiefelLayer(
            6, 8, 7, 2)
        @test_throws r"^(?:ArgumentError: )?mv must satisfy 1 ≤ mv ≤ Nv ÷ 2; got mv = 0, Nv = 8$" BilinearStiefelLayer(
            6, 8, 3, 0)
        @test_throws r"^(?:ArgumentError: )?mv must satisfy 1 ≤ mv ≤ Nv ÷ 2; got mv = 5, Nv = 8$" BilinearStiefelLayer(
            6, 8, 3, 5)
        @test_throws r"^(?:ArgumentError: )?metric_factors must be a 2-tuple of matrices \(Rx, Rv\) or nothing$" BilinearStiefelLayer(
            6, 8, 3, 2; metric_factors = (ones(T, 6), ones(T, 4)))
        @test_throws r"^(?:DimensionMismatch: )?Rx must have size \(6, 6\); got \(5, 5\)$" BilinearStiefelLayer(
            6, 8, 3, 2; metric_factors = (Matrix{T}(I, 5, 5), Matrix{T}(I, 4, 4)))
        @test_throws r"^(?:DimensionMismatch: )?Rv must have size \(4, 4\) or \(8, 8\); got \(3, 3\)$" BilinearStiefelLayer(
            6, 8, 3, 2; metric_factors = (Matrix{T}(I, 6, 6), Matrix{T}(I, 3, 3)))

        # TransposeBilinearStiefelLayer constructor guards
        @test_throws r"^(?:ArgumentError: )?Nv must be an even integer ≥ 2; got Nv = 5$" TransposeBilinearStiefelLayer(
            6, 5, 2, 1)
        @test_throws r"^(?:ArgumentError: )?mx must satisfy 1 ≤ mx ≤ Nx; got mx = 8, Nx = 6$" TransposeBilinearStiefelLayer(
            6, 8, 8, 2)

        # Initial parameters and manifold constraints
        ps = @inferred initialparameters(
            rng, GlorotUniform(), l, CPU(), T)
        @test haskey(ps, :Φx) && haskey(ps, :Φv)
        @test ps.Φx isa StiefelManifold{T}
        @test ps.Φv isa StiefelManifold{T}
        @test size(ps.Φx) == (Nx, mx)
        @test size(ps.Φv) == (Nv ÷ 2, mv)
        @test norm(ps.Φx' * ps.Φx - I(mx)) <= 20 * eps(T)
        @test norm(ps.Φv' * ps.Φv - I(mv)) <= 20 * eps(T)

        ps_inv = @inferred initialparameters(
            rng, GlorotUniform(), inv_l, CPU(), T)
        @test size(ps_inv.Φx) == (Nx, mx)
        @test size(ps_inv.Φv) == (Nv ÷ 2, mv)

        expected_len = (Nx * mx - (mx * (mx + 1)) ÷ 2) +
                       ((Nv ÷ 2) * mv - (mv * (mv + 1)) ÷ 2)
        @test parameterlength(l) == expected_len
        @test parameterlength(inv_l) == expected_len
    end
end

@testset "BilinearStiefelLayer forward, inverse_layer, parity isometry, Kronecker symplecticity" begin
    for T in REAL_ELTYPES
        rng = MersenneTwister(101)
        Nx, Nv, mx, mv = 6, 8, 3, 2
        n_lat = mx * mv
        l = BilinearStiefelLayer(Nx, Nv, mx, mv)
        inv_l = inverse_layer(l)
        ps = initialparameters(
            rng, GlorotUniform(), l, CPU(), T)

        # 1. Parity split isometry ||F^+||_F^2 + ||F^-||_F^2 == ||F||_F^2 to 20*eps(T)
        F = randn(rng, T, Nx, Nv)
        F_plus, F_minus = GeometricMachineLearning._parity_split(F)
        inv_sqrt2 = inv(sqrt(T(2)))
        @test norm(F_plus - inv_sqrt2 * (F[:, 1:(Nv ÷ 2)] + F[:, Nv:-1:(Nv ÷ 2 + 1)])) <=
              10 * eps(T)
        @test norm(F_minus - inv_sqrt2 * (F[:, 1:(Nv ÷ 2)] - F[:, Nv:-1:(Nv ÷ 2 + 1)])) <=
              10 * eps(T)
        @test abs((norm(F_plus)^2 + norm(F_minus)^2) - norm(F)^2) <= 20 * eps(T) * norm(F)^2

        # Full-rank layer preserves total Frobenius norm ||q||^2 + ||p||^2 == ||F||_F^2 (catches M2)
        l_full = BilinearStiefelLayer(Nx, Nv, Nx, Nv ÷ 2)
        ps_full = initialparameters(
            rng, GlorotUniform(), l_full, CPU(), T)
        qp_full = @inferred l_full(F, ps_full)
        @test abs((norm(qp_full.q)^2 + norm(qp_full.p)^2) - norm(F)^2) <=
              20 * eps(T) * norm(F)^2
        F_full_rec = @inferred inverse_layer(l_full)(qp_full, ps_full)
        @test norm(F_full_rec - F) <= 30 * eps(T) * norm(F)

        # Purely v-even and v-odd fields under v -> -v reversal (catches M1)
        A_half = randn(rng, T, Nx, Nv ÷ 2)
        # Ensure A_half[:, j] != A_half[:, end + 1 - j]
        F_even = hcat(A_half, A_half[:, end:-1:1])
        F_odd = hcat(A_half, -A_half[:, end:-1:1])
        qp_even = @inferred l(F_even, ps)
        qp_odd = @inferred l(F_odd, ps)
        @test norm(qp_even.p) <= 20 * eps(T) * norm(F_even)
        @test norm(qp_odd.q) <= 20 * eps(T) * norm(F_odd)

        # Pre-split (q = F_plus, p = F_minus) input produces identical latent (q, p)
        qp_from_F = @inferred l(F, ps)
        qp_from_split = @inferred l((q = F_plus, p = F_minus), ps)
        @test length(qp_from_F.q) == n_lat
        @test length(qp_from_F.p) == n_lat
        @test norm(qp_from_F.q - qp_from_split.q) <= 20 * eps(T)
        @test norm(qp_from_F.p - qp_from_split.p) <= 20 * eps(T)

        # Right-inverse identity: l(inverse_layer(l)((q = q, p = p), ps), ps) == (q = q, p = p)
        q_lat = randn(rng, T, n_lat)
        p_lat = randn(rng, T, n_lat)
        F_decoded = @inferred inv_l((q = q_lat, p = p_lat), ps)
        @test size(F_decoded) == (Nx, Nv)
        qp_roundtrip = @inferred l(F_decoded, ps)
        @test norm(qp_roundtrip.q - q_lat) <= 30 * eps(T) * max(one(T), norm(q_lat))
        @test norm(qp_roundtrip.p - p_lat) <= 30 * eps(T) * max(one(T), norm(p_lat))

        # Matrix-shaped (Q_red, P_red) tuple input to decoder
        Q_mat = reshape(q_lat, mx, mv)
        P_mat = reshape(p_lat, mx, mv)
        F_decoded_tuple = @inferred inv_l((Q_mat, P_mat), ps)
        @test norm(F_decoded_tuple - F_decoded) <= 10 * eps(T)

        # 3D batched forward and inverse passes
        B = 5
        F_batch = randn(rng, T, Nx, Nv, B)
        qp_batch = @inferred l(F_batch, ps)
        @test size(qp_batch.q) == (n_lat, B)
        @test size(qp_batch.p) == (n_lat, B)
        for b in 1:B
            qp_b = l(@view(F_batch[:, :, b]), ps)
            @test norm(qp_batch.q[:, b] - qp_b.q) <= 20 * eps(T)
            @test norm(qp_batch.p[:, b] - qp_b.p) <= 20 * eps(T)
        end
        F_plus_b, F_minus_b = GeometricMachineLearning._parity_split(F_batch)
        qp_batch_split = @inferred l((q = F_plus_b, p = F_minus_b), ps)
        @test norm(qp_batch_split.q - qp_batch.q) <= 20 * eps(T)
        @test norm(qp_batch_split.p - qp_batch.p) <= 20 * eps(T)

        q_lat_b = randn(rng, T, n_lat, B)
        p_lat_b = randn(rng, T, n_lat, B)
        F_dec_b = @inferred inv_l((q = q_lat_b, p = p_lat_b), ps)
        @test size(F_dec_b) == (Nx, Nv, B)
        F_dec_b3 = @inferred inv_l(
            (q = reshape(q_lat_b, mx, mv, B), p = reshape(p_lat_b, mx, mv, B)), ps)
        @test norm(F_dec_b3 - F_dec_b) <= 10 * eps(T)
        qp_rt_b = @inferred l(F_dec_b, ps)
        @test norm(qp_rt_b.q - q_lat_b) <= 30 * eps(T) * norm(q_lat_b)
        @test norm(qp_rt_b.p - p_lat_b) <= 30 * eps(T) * norm(p_lat_b)

        # Kronecker symplecticity identity:
        # diag(Φ_kron, Φ_kron)^T * J_{Nx*Nv} * diag(Φ_kron, Φ_kron) == J_{2*mx*mv}
        Φx_mat = Matrix{T}(ps.Φx)
        Φv_mat = Matrix{T}(ps.Φv)
        Φ_kron = kron(Φv_mat, Φx_mat)
        @test size(Φ_kron) == (Nx * (Nv ÷ 2), n_lat)
        @test norm(qp_from_F.q - Φ_kron' * vec(F_plus)) <= 20 * eps(T) * norm(F)
        @test norm(qp_from_F.p - Φ_kron' * vec(F_minus)) <= 20 * eps(T) * norm(F)
        Z_blk = zeros(T, Nx * (Nv ÷ 2), n_lat)
        L_kron = [Φ_kron Z_blk; Z_blk Φ_kron]
        J_full = _canonical_symplectic_matrix(T, Nx * Nv)
        J_red = _canonical_symplectic_matrix(T, 2 * n_lat)
        @test norm(L_kron' * J_full * L_kron - J_red) <= 30 * eps(T)

        # Metric factors (half-velocity and full-velocity)
        Rx = Diagonal(T[1.2, 0.8, 1.5, 0.9, 1.1, 1.3]) |> Matrix
        Rv_half = Diagonal(T[0.7, 1.4, 0.9, 1.2]) |> Matrix
        l_w = BilinearStiefelLayer(Nx, Nv, mx, mv; metric_factors = (Rx, Rv_half))
        inv_lw = @inferred inverse_layer(l_w)
        F_dec_w = @inferred inv_lw((q = q_lat, p = p_lat), ps)
        qp_rt_w = @inferred l_w(F_dec_w, ps)
        @test norm(qp_rt_w.q - q_lat) <= 30 * eps(T) * max(one(T), norm(q_lat))
        @test norm(qp_rt_w.p - p_lat) <= 30 * eps(T) * max(one(T), norm(p_lat))

        Rv_full = Diagonal(vcat(diag(Rv_half), reverse(diag(Rv_half)))) |> Matrix
        l_wfull = BilinearStiefelLayer(
            Nx, Nv, mx, mv; metric_factors = (Rx, Rv_full))
        inv_lwfull = @inferred inverse_layer(l_wfull)
        F_dec_wf = @inferred inv_lwfull((q = q_lat, p = p_lat), ps)
        qp_rt_wf = @inferred l_wfull(F_dec_wf, ps)
        @test norm(qp_rt_wf.q - q_lat) <= 30 * eps(T) * max(one(T), norm(q_lat))
        @test norm(qp_rt_wf.p - p_lat) <= 30 * eps(T) * max(one(T), norm(p_lat))

        # DimensionMismatch guards on forward and inverse calls
        @test_throws r"^(?:DimensionMismatch: )?F must have first two dimensions \(6, 8\); got size\(F\) = \(5, 8\)$" l(
            zeros(T, 5, 8), ps)
        @test_throws r"^(?:DimensionMismatch: )?q and p must have the same size; got \(6, 4\) and \(6, 3\)$" l(
            (q = zeros(T, 6, 4), p = zeros(T, 6, 3)), ps)
        @test_throws r"^(?:DimensionMismatch: )?pre-split q and p must have first two dimensions \(6, 4\); got \(6, 5\)$" l(
            (q = zeros(T, 6, 5), p = zeros(T, 6, 5)), ps)
        @test_throws r"^(?:DimensionMismatch: )?latent \(q, p\) must have size \(6,\) or \(3, 2\) \(or batched\); got \(5,\)$" inv_l(
            (q = zeros(T, 5), p = zeros(T, 5)), ps)
        @test_throws r"^(?:DimensionMismatch: )?q and p must have the same size; got \(6,\) and \(5,\)$" inv_l(
            (q = zeros(T, 6), p = zeros(T, 5)), ps)

        # Hostile inputs: zeros, subnormals, large values, minimum dimensions Nx=1, Nv=2, mx=1, mv=1
        l_min = BilinearStiefelLayer(1, 2, 1, 1)
        ps_min = initialparameters(
            rng, GlorotUniform(), l_min, CPU(), T)
        for scale in (zero(T), floatmin(T), T(1e2))
            F_h = fill(scale, 1, 2)
            qp_h = @inferred l_min(F_h, ps_min)
            @test all(isfinite, qp_h.q) && all(isfinite, qp_h.p)
        end
    end
end

@testset "LowRankStiefelLayer and DecodeLowRankStiefelLayer" begin
    for T in REAL_ELTYPES
        rng = MersenneTwister(202)
        Nx, Nv, mx, mv = 8, 10, 4, 3
        r_max = min(mx, 2 * mv)
        l_dlra = LowRankStiefelLayer(Nx, Nv, mx, mv)
        inv_dlra = @inferred inverse_layer(l_dlra)
        @test inv_dlra isa DecodeLowRankStiefelLayer
        @test (@inferred inverse_layer(inv_dlra)) === l_dlra

        l_grid = BilinearStiefelLayer(Nx, Nv, mx, mv)
        inv_grid = inverse_layer(l_grid)
        ps = @inferred initialparameters(
            rng, GlorotUniform(), l_dlra, CPU(), T)
        @test parameterlength(l_dlra) == parameterlength(l_grid)
        @test parameterlength(inv_dlra) == parameterlength(l_grid)

        # Constructor guards
        @test_throws r"^(?:ArgumentError: )?Nx must be ≥ 1; got Nx = 0$" LowRankStiefelLayer(0, 10, 1, 1)
        @test_throws r"^(?:ArgumentError: )?Nv must be an even integer ≥ 2; got Nv = 9$" LowRankStiefelLayer(
            8, 9, 2, 2)
        @test_throws r"^(?:ArgumentError: )?mx must satisfy 1 ≤ mx ≤ Nx; got mx = 9, Nx = 8$" LowRankStiefelLayer(
            8, 10, 9, 2)
        @test_throws r"^(?:ArgumentError: )?mx must satisfy 1 ≤ mx ≤ Nx; got mx = 0, Nx = 8$" LowRankStiefelLayer(
            8, 10, 0, 2)
        @test_throws r"^(?:ArgumentError: )?mv must satisfy 1 ≤ mv ≤ Nv ÷ 2; got mv = 6, Nv = 10$" LowRankStiefelLayer(
            8, 10, 4, 6)
        @test_throws r"^(?:ArgumentError: )?mv must satisfy 1 ≤ mv ≤ Nv ÷ 2; got mv = 0, Nv = 10$" LowRankStiefelLayer(
            8, 10, 4, 0)
        @test_throws r"^(?:ArgumentError: )?r_out must be ≥ 1; got r_out = 0$" LowRankStiefelLayer(
            8, 10, 4, 3; r_out = 0)
        @test_throws r"^(?:ArgumentError: )?Nv must be an even integer ≥ 2; got Nv = 3$" DecodeLowRankStiefelLayer(
            8, 3, 2, 1)
        @test_throws r"^(?:ArgumentError: )?r_out must be ≥ 1; got r_out = -1$" DecodeLowRankStiefelLayer(
            8, 10, 4, 3; r_out = -1)

        # DimensionMismatch guards on (X, S, V)
        @test_throws r"^(?:DimensionMismatch: )?X must have 8 rows; got size\(X\) = \(7, 3\)$" l_dlra(
            (X = zeros(T, 7, 3), S = zeros(T, 3, 3), V = zeros(T, 10, 3)), ps)
        @test_throws r"^(?:DimensionMismatch: )?DLRA rank r_in must be ≥ 1; got r_in = 0$" l_dlra(
            (X = zeros(T, 8, 0), S = zeros(T, 0, 0), V = zeros(T, 10, 0)), ps)
        @test_throws r"^(?:DimensionMismatch: )?S must have size \(3, 3\); got size\(S\) = \(2, 3\)$" l_dlra(
            (X = zeros(T, 8, 3), S = zeros(T, 2, 3), V = zeros(T, 10, 3)), ps)
        @test_throws r"^(?:DimensionMismatch: )?V must have size \(10, 3\); got size\(V\) = \(10, 2\)$" l_dlra(
            (X = zeros(T, 8, 3), S = zeros(T, 3, 3), V = zeros(T, 10, 2)), ps)
        @test_throws r"^(?:DimensionMismatch: )?V must have size \(10, 3\); got size\(V\) = \(8, 3\)$" l_dlra(
            (X = zeros(T, 8, 3), S = zeros(T, 3, 3), V = zeros(T, 8, 3)), ps)
        @test_throws r"^(?:DimensionMismatch: )?S must have size \(2, 2, 4\); got size\(S\) = \(2, 2, 3\)$" l_dlra(
            (X = zeros(T, 8, 2, 4), S = zeros(T, 2, 2, 3), V = zeros(T, 10, 2, 4)), ps)
        @test_throws r"^(?:DimensionMismatch: )?V must have size \(10, 2, 4\); got size\(V\) = \(10, 2, 3\)$" l_dlra(
            (X = zeros(T, 8, 2, 4), S = zeros(T, 2, 2, 4), V = zeros(T, 10, 2, 3)), ps)

        # Arbitrary r_in >= 1: exact cross-modal equivalence with BilinearStiefelLayer and O(r_in)xO(r_in) gauge invariance
        for r_in in (1, 3, 6)
            X = _random_stiefel(rng, T, Nx, r_in)
            S = randn(rng, T, r_in, r_in)
            V = _random_stiefel(rng, T, Nv, r_in)
            F_dense = X * S * V'

            qp_dlra = @inferred l_dlra((X = X, S = S, V = V), ps)
            qp_grid = @inferred l_grid(F_dense, ps)
            @test norm(qp_dlra.q - qp_grid.q) <= 50 * eps(T) * max(one(T), norm(F_dense))
            @test norm(qp_dlra.p - qp_grid.p) <= 50 * eps(T) * max(one(T), norm(F_dense))

            # Gauge transformation (X*Ox, Ox'*S*Ov, V*Ov) including improper reflection
            Ox = _random_stiefel(rng, T, r_in, r_in)
            Ov = _random_stiefel(rng, T, r_in, r_in)
            Ov[:, 1] .*= -one(T)
            qp_gauge = @inferred l_dlra((X = X * Ox, S = Ox' * S * Ov, V = V * Ov), ps)
            @test norm(qp_gauge.q - qp_dlra.q) <= 50 * eps(T) * max(one(T), norm(F_dense))
            @test norm(qp_gauge.p - qp_dlra.p) <= 50 * eps(T) * max(one(T), norm(F_dense))
        end

        # Decoder DecodeLowRankStiefelLayer: Stiefel membership and consistency with TransposeBilinearStiefelLayer
        n_lat = mx * mv
        q_lat = randn(rng, T, n_lat)
        p_lat = randn(rng, T, n_lat)
        decoded = @inferred inv_dlra((q = q_lat, p = p_lat), ps)
        @test haskey(decoded, :X) && haskey(decoded, :S) && haskey(decoded, :V)
        @test size(decoded.X) == (Nx, r_max)
        @test size(decoded.S) == (r_max, r_max)
        @test size(decoded.V) == (Nv, r_max)
        @test norm(decoded.X' * decoded.X - I(r_max)) <= 30 * eps(T)
        @test norm(decoded.V' * decoded.V - I(r_max)) <= 30 * eps(T)

        F_from_dlra = decoded.X * decoded.S * decoded.V'
        F_from_grid = inv_grid((q = q_lat, p = p_lat), ps)
        @test norm(F_from_dlra - F_from_grid) <=
              50 * eps(T) * max(one(T), norm(F_from_grid))

        # Round-trip encode(decode((q, p))) == (q, p) when r_out = min(mx, 2*mv)
        qp_rt = @inferred l_dlra(decoded, ps)
        @test norm(qp_rt.q - q_lat) <= 50 * eps(T) * max(one(T), norm(q_lat))
        @test norm(qp_rt.p - p_lat) <= 50 * eps(T) * max(one(T), norm(p_lat))

        # Truncated r_out < min(mx, 2*mv)
        r_trunc = 2
        l_trunc = LowRankStiefelLayer(Nx, Nv, mx, mv; r_out = r_trunc)
        inv_trunc = inverse_layer(l_trunc)
        dec_trunc = @inferred inv_trunc((q = q_lat, p = p_lat), ps)
        @test size(dec_trunc.X) == (Nx, r_trunc)
        @test size(dec_trunc.S) == (r_trunc, r_trunc)
        @test size(dec_trunc.V) == (Nv, r_trunc)
        @test norm(dec_trunc.X' * dec_trunc.X - I(r_trunc)) <= 30 * eps(T)
        @test norm(dec_trunc.V' * dec_trunc.V - I(r_trunc)) <= 30 * eps(T)

        # Weighted metric_factors on LowRankStiefelLayer and DecodeLowRankStiefelLayer (catches M3!)
        Rx = Diagonal(T.(collect(range(T(0.7), T(1.5); length = Nx)))) |> Matrix
        Rv_half = Diagonal(T.(collect(range(T(0.6), T(1.6); length = Nv ÷ 2)))) |> Matrix
        l_dlra_w = LowRankStiefelLayer(
            Nx, Nv, mx, mv; r_out = r_max, metric_factors = (Rx, Rv_half))
        inv_dlra_w = @inferred inverse_layer(l_dlra_w)
        l_grid_w = BilinearStiefelLayer(Nx, Nv, mx, mv; metric_factors = (Rx, Rv_half))
        inv_grid_w = inverse_layer(l_grid_w)

        X_w = _random_stiefel(rng, T, Nx, 3)
        S_w = randn(rng, T, 3, 3)
        V_w = _random_stiefel(rng, T, Nv, 3)
        qp_dlra_w = @inferred l_dlra_w((X = X_w, S = S_w, V = V_w), ps)
        qp_grid_w = @inferred l_grid_w(X_w * S_w * V_w', ps)
        @test norm(qp_dlra_w.q - qp_grid_w.q) <= 50 * eps(T) * max(one(T), norm(S_w))
        @test norm(qp_dlra_w.p - qp_grid_w.p) <= 50 * eps(T) * max(one(T), norm(S_w))

        dec_w = @inferred inv_dlra_w((q = q_lat, p = p_lat), ps)
        @test norm(dec_w.X' * dec_w.X - I(r_max)) <= 30 * eps(T)
        @test norm(dec_w.V' * dec_w.V - I(r_max)) <= 30 * eps(T)
        F_grid_w = inv_grid_w((q = q_lat, p = p_lat), ps)
        @test norm(dec_w.X * dec_w.S * dec_w.V' - F_grid_w) <=
              50 * eps(T) * max(one(T), norm(F_grid_w))

        # Also test non-orthonormal Φv perturbation on unweighted DecodeLowRankStiefelLayer (catches M3 even on unweighted path)
        ps_scaled = (Φx = ps.Φx, Φv = T(1.3) .* Matrix{T}(ps.Φv))
        dec_scaled = @inferred inv_dlra((q = q_lat, p = p_lat), ps_scaled)
        @test norm(dec_scaled.X' * dec_scaled.X - I(r_max)) <= 30 * eps(T)
        @test norm(dec_scaled.V' * dec_scaled.V - I(r_max)) <= 30 * eps(T)

        # Batched 3D DLRA encode and decode
        B = 3
        X_b = zeros(T, Nx, 2, B)
        S_b = randn(rng, T, 2, 2, B)
        V_b = zeros(T, Nv, 2, B)
        for b in 1:B
            X_b[:, :, b] = _random_stiefel(rng, T, Nx, 2)
            V_b[:, :, b] = _random_stiefel(rng, T, Nv, 2)
        end
        qp_b = @inferred l_dlra((X = X_b, S = S_b, V = V_b), ps)
        @test size(qp_b.q) == (n_lat, B)
        @test size(qp_b.p) == (n_lat, B)
        dec_b = @inferred inv_dlra(qp_b, ps)
        @test size(dec_b.X) == (Nx, r_max, B)
        @test size(dec_b.S) == (r_max, r_max, B)
        @test size(dec_b.V) == (Nv, r_max, B)
        for b in 1:B
            @test norm(dec_b.X[:, :, b]' * dec_b.X[:, :, b] - I(r_max)) <= 30 * eps(T)
            @test norm(dec_b.V[:, :, b]' * dec_b.V[:, :, b] - I(r_max)) <= 30 * eps(T)
        end
    end
end

@testset "FastDLRATraceLoss" begin
    for T in REAL_ELTYPES
        rng = MersenneTwister(303)
        Nx, Nv = 10, 12
        r1, r2 = 3, 4
        loss = @inferred FastDLRATraceLoss()

        # 1. Exact Stiefel factors with distinct ranks r1 != r2
        X1 = _random_stiefel(rng, T, Nx, r1)
        S1 = randn(rng, T, r1, r1)
        V1 = _random_stiefel(rng, T, Nv, r1)
        X2 = _random_stiefel(rng, T, Nx, r2)
        S2 = randn(rng, T, r2, r2)
        V2 = _random_stiefel(rng, T, Nv, r2)

        F1 = X1 * S1 * V1'
        F2 = X2 * S2 * V2'
        dense_sq = norm(F1 - F2)^2
        fast_sq = @inferred loss((X = X1, S = S1, V = V1), (X = X2, S = S2, V = V2))
        @test fast_sq isa T
        @test abs(fast_sq - dense_sq) <= 100 * eps(T) * norm(F1)^2

        # 2. Approximately orthonormal factors ||X^T X - I||_F ~ 0.1 (catches M4!)
        X1_approx = X1 + T(0.05) * randn(rng, T, Nx, r1)
        V1_approx = V1 + T(0.05) * randn(rng, T, Nv, r1)
        X2_approx = X2 + T(0.05) * randn(rng, T, Nx, r2)
        V2_approx = V2 + T(0.05) * randn(rng, T, Nv, r2)
        @test norm(X1_approx' * X1_approx - I(r1)) > T(0.02)
        F1_approx = X1_approx * S1 * V1_approx'
        F2_approx = X2_approx * S2 * V2_approx'
        dense_approx_sq = norm(F1_approx - F2_approx)^2
        fast_approx_sq = @inferred loss(
            (X = X1_approx, S = S1, V = V1_approx),
            (X = X2_approx, S = S2, V = V2_approx))
        @test abs(fast_approx_sq - dense_approx_sq) <= 100 * eps(T) * norm(F1_approx)^2

        # 3. Non-negativity clamp and gauge invariance at zero error
        Ox = _random_stiefel(rng, T, r1, r1)
        Ov = _random_stiefel(rng, T, r1, r1)
        self_loss = @inferred loss((X = X1, S = S1, V = V1), (
            X = X1 * Ox, S = Ox' * S1 * Ov, V = V1 * Ov))
        @test self_loss >= zero(T)
        @test self_loss <= 100 * eps(T) * norm(F1)^2

        # 4. Batched 3D DLRA loss and NetworkLoss interface
        B = 3
        Xb1 = repeat(X1_approx, 1, 1, B)
        Sb1 = repeat(S1, 1, 1, B)
        Vb1 = repeat(V1_approx, 1, 1, B)
        Xb2 = repeat(X2_approx, 1, 1, B)
        Sb2 = repeat(S2, 1, 1, B)
        Vb2 = repeat(V2_approx, 1, 1, B)
        fast_batch_sq = @inferred loss((X = Xb1, S = Sb1, V = Vb1), (
            X = Xb2, S = Sb2, V = Vb2))
        @test abs(fast_batch_sq - B * dense_approx_sq) <=
              100 * eps(T) * B * norm(F1_approx)^2

        # NetworkLoss interface with Chain(LowRankStiefelLayer, DecodeLowRankStiefelLayer)
        l_dlra = LowRankStiefelLayer(Nx, Nv, 3, 2; r_out = 3)
        chain = Chain(l_dlra, inverse_layer(l_dlra))
        ps_chain = initialparameters(
            rng, GlorotUniform(), chain, CPU(), T)
        dlra_in = (X = X1, S = S1, V = V1)
        dlra_out = chain(dlra_in, ps_chain)
        loss_net = @inferred loss(chain, ps_chain, dlra_in, dlra_in)
        @test abs(loss_net - loss(dlra_out, dlra_in)) <= 20 * eps(T) * norm(F1)^2

        # DimensionMismatch guards
        @test_throws r"^(?:DimensionMismatch: )?X and X_hat must have the same number of rows; got 10 and 9$" loss(
            (X = X1, S = S1, V = V1),
            (X = zeros(T, 9, r2), S = S2, V = V2))
        @test_throws r"^(?:DimensionMismatch: )?V and V_hat must have the same number of rows; got 12 and 11$" loss(
            (X = X1, S = S1, V = V1),
            (X = X2, S = S2, V = zeros(T, 11, r2)))
    end
end

@testset "solve! Tucker-2 HOSVD initialization on BilinearStiefelLayer" begin
    for T in REAL_ELTYPES
        rng = MersenneTwister(404)
        Nx, Nv, mx, mv = 8, 10, 3, 2
        n_snaps = 15

        # Construct synthetic exact rank-(mx, mv) snapshots in the parity basis
        Φx_true = _random_stiefel(rng, T, Nx, mx)
        Φv_true = _random_stiefel(rng, T, Nv ÷ 2, mv)
        F_snapshots = zeros(T, Nx, Nv, n_snaps)
        for k in 1:n_snaps
            Q_k = randn(rng, T, mx, mv)
            P_k = randn(rng, T, mx, mv)
            F_plus_k = Φx_true * Q_k * Φv_true'
            F_minus_k = Φx_true * P_k * Φv_true'
            F_snapshots[:, :, k] = GeometricMachineLearning._parity_recombine(F_plus_k, F_minus_k)
        end

        l = BilinearStiefelLayer(Nx, Nv, mx, mv)
        inv_l = inverse_layer(l)
        nn = NeuralNetwork(Chain(l, inv_l), CPU(), T)
        solve!(nn, F_snapshots)

        ps_svd = nn.params.L1
        @test norm(ps_svd.Φx' * ps_svd.Φx - I(mx)) <= 30 * eps(T)
        @test norm(ps_svd.Φv' * ps_svd.Φv - I(mv)) <= 30 * eps(T)
        # Decoder parameters in Chain(l, inv_l) also updated
        @test norm(Matrix{T}(nn.params.L2.Φx) - Matrix{T}(ps_svd.Φx)) <= 10 * eps(T)
        @test norm(Matrix{T}(nn.params.L2.Φv) - Matrix{T}(ps_svd.Φv)) <= 10 * eps(T)

        # Subspace projectors match Φx_true and Φv_true to 50*eps(T)
        Px_est = Matrix{T}(ps_svd.Φx) * Matrix{T}(ps_svd.Φx)'
        Px_true = Φx_true * Φx_true'
        Pv_est = Matrix{T}(ps_svd.Φv) * Matrix{T}(ps_svd.Φv)'
        Pv_true = Φv_true * Φv_true'
        @test norm(Px_est - Px_true) <= 50 * eps(T)
        @test norm(Pv_est - Pv_true) <= 50 * eps(T)

        # Exact reconstruction of rank-(mx, mv) snapshots to 50*eps(T)
        F_rec = inv_l(l(F_snapshots, ps_svd), ps_svd)
        @test norm(F_rec - F_snapshots) <= 50 * eps(T) * norm(F_snapshots)

        # Also test solve!(l, ps, F_snapshots) and weighted metric_factors
        Rx = Diagonal(T.(collect(range(T(0.8), T(1.4); length = Nx)))) |> Matrix
        Rv_half = Diagonal(T.(collect(range(T(0.7), T(1.3); length = Nv ÷ 2)))) |> Matrix
        l_w = BilinearStiefelLayer(Nx, Nv, mx, mv; metric_factors = (Rx, Rv_half))
        inv_lw = inverse_layer(l_w)
        F_w_snaps = zeros(T, Nx, Nv, n_snaps)
        for k in 1:n_snaps
            F_plus_k, F_minus_k = GeometricMachineLearning._parity_split(@view(F_snapshots[:, :, k]))
            F_w_snaps[:, :, k] = GeometricMachineLearning._parity_recombine(
                Rx \ F_plus_k / Rv_half', Rx \ F_minus_k / Rv_half')
        end
        nn_w = NeuralNetwork(Chain(l_w), CPU(), T)
        solve!(nn_w, F_w_snaps)
        F_w_rec = inv_lw(l_w(F_w_snaps, nn_w.params.L1), nn_w.params.L1)
        @test norm(F_w_rec - F_w_snaps) <= 50 * eps(T) * norm(F_w_snaps)

        # DimensionMismatch guard on solve!
        @test_throws r"^(?:DimensionMismatch: )?F_snapshots must have first two dimensions \(8, 10\) and at least 1 snapshot; got \(7, 10, 15\)$" solve!(
            nn, zeros(T, 7, 10, 15))
    end
end

@testset "Batched Zygote differentiation and allocation scaling" begin
    for T in REAL_ELTYPES
        rng = MersenneTwister(505)
        Nx, Nv, mx, mv = 6, 8, 3, 2
        B = 3
        l = BilinearStiefelLayer(Nx, Nv, mx, mv)
        inv_l = inverse_layer(l)
        ps = initialparameters(
            rng, GlorotUniform(), l, CPU(), T)
        ps_nt = (Φx = Matrix{T}(ps.Φx), Φv = Matrix{T}(ps.Φv))

        F_b = randn(rng, T, Nx, Nv, B)
        gs = Zygote.gradient(F_in -> sum(abs2, inv_l(l(F_in, ps_nt), ps_nt)), F_b)[1]
        @test size(gs) == size(F_b)
        @test all(isfinite, gs)

        # Finite-difference check on one entry in Float64
        if T === Float64
            eps_fd = 1e-6
            F_p = copy(F_b)
            F_p[2, 3, 1] += eps_fd
            F_m = copy(F_b)
            F_m[2, 3, 1] -= eps_fd
            fd_grad = (sum(abs2, inv_l(l(F_p, ps_nt), ps_nt)) -
                       sum(abs2, inv_l(l(F_m, ps_nt), ps_nt))) / (2 * eps_fd)
            @test isapprox(gs[2, 3, 1], fd_grad; rtol = 1e-5, atol = 1e-7)
        end

        # Zygote gradient through LowRankStiefelLayer + FastDLRATraceLoss
        l_dlra = LowRankStiefelLayer(Nx, Nv, mx, mv)
        X = _random_stiefel(rng, T, Nx, 2)
        S = randn(rng, T, 2, 2)
        V = _random_stiefel(rng, T, Nv, 2)
        gs_dlra = Zygote.gradient(
            S_in -> begin
                qp = l_dlra((X = X, S = S_in, V = V), ps_nt)
                sum(abs2, qp.q) + sum(abs2, qp.p)
            end, S)[1]
        @test size(gs_dlra) == (2, 2)
        @test all(isfinite, gs_dlra)

        loss = FastDLRATraceLoss()
        gs_loss = Zygote.gradient(
            S_in -> loss((X = X, S = S_in, V = V), (
                X = X, S = 2 .* S, V = V)), S)[1]
        @test size(gs_loss) == (2, 2)
        @test all(isfinite, gs_loss)

        # Two-batch-size allocation scaling (B1 = 4 vs B2 = 32)
        B1, B2 = 4, 32
        F_b1 = randn(rng, T, Nx, Nv, B1)
        F_b2 = randn(rng, T, Nx, Nv, B2)
        l(F_b1, ps_nt)
        l(F_b2, ps_nt)
        alloc_l1 = @allocated l(F_b1, ps_nt)
        alloc_l2 = @allocated l(F_b2, ps_nt)
        @test abs(alloc_l2 / B2 - alloc_l1 / B1) <= 1024

        dlra_b1 = (
            X = repeat(X, 1, 1, B1), S = repeat(S, 1, 1, B1), V = repeat(V, 1, 1, B1))
        dlra_b2 = (
            X = repeat(X, 1, 1, B2), S = repeat(S, 1, 1, B2), V = repeat(V, 1, 1, B2))
        loss(dlra_b1, dlra_b1)
        loss(dlra_b2, dlra_b2)
        alloc_loss1 = @allocated loss(dlra_b1, dlra_b1)
        alloc_loss2 = @allocated loss(dlra_b2, dlra_b2)
        @test abs(alloc_loss2 / B2 - alloc_loss1 / B1) <= 512
    end
end
