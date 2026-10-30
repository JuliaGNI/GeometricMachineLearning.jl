using GeometricMachineLearning
using GeometricMachineLearning: HRedSys, build_v_reduced, build_f_reduced,
                                integrate_reduced_system, integrate_full_system,
                                evaluate_vf_and_compute_∇Ψ,
                                evaluate_vf_and_vjp, ReducedVFCache,
                                ForwardDiff, StateVariable
using GeometricIntegrators: ImplicitMidpoint
using KernelAbstractions
using LinearAlgebra
using Test
using Random: Random

include("../helpers/eltypes.jl")

Random.seed!(42)

@testset "HRedSys reference-state shifting, pullback_mode = :jacobian | :vjp, and work buffers" begin
    model_chk = WeightTiedSymplecticAutoencoder(
        6, 2; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)
    nn_chk = NeuralNetwork(model_chk, CPU(), Float64)
    v_dummy!(v, t, q, p, ps) = (v .= p; nothing)
    f_dummy!(f, t, q, p, ps) = (f .= .-q; nothing)
    h_dummy(t, q, p, ps) = 0.5 * (sum(abs2, q) + sum(abs2, p))

    re_mode = r"^(?:ArgumentError: )?pullback_mode must be :jacobian or :vjp, got :invalid\.$"
    @test_throws re_mode build_v_reduced(
        v_dummy!, f_dummy!, decoder(nn_chk); pullback_mode = :invalid)
    @test_throws re_mode build_f_reduced(
        v_dummy!, f_dummy!, decoder(nn_chk); pullback_mode = :invalid)

    re_ref_dim = r"^(?:DimensionMismatch: )?q_ref and p_ref must have length N = 3, got 2 and 3\.$"
    ics_chk = (q = StateVariable(zeros(3)), p = StateVariable(zeros(3)))
    @test_throws re_ref_dim HRedSys(
        6, 2, encoder(nn_chk), decoder(nn_chk), v_dummy!, f_dummy!, h_dummy,
        (0.0, 0.1), 0.05, ics_chk;
        integrator = ImplicitMidpoint(),
        q_ref = zeros(2),
        p_ref = zeros(3)
    )

    @testset "2N = 12, 2n = 4 VJP vs Jacobian agreement and equilibrium integration ($T)" for T in REAL_ELTYPES
        rng = Random.MersenneTwister(404)
        N = 6
        n = 2
        full_dim = 2 * N # 12
        reduced_dim = 2 * n # 4

        arch = WeightTiedSymplecticAutoencoder(
            full_dim, reduced_dim;
            n_blocks = 1,
            n_layers = 1,
            n_ambient_layers = 0,
            sympnet_upscale = 2
        )
        nn = NeuralNetwork(arch, CPU(), T; rng = rng)

        q_ref = randn(rng, T, N)
        p_ref = randn(rng, T, N)

        # Shifted quadratic + anharmonic Hamiltonian vector field:
        # H(q, p) = 1/2 ||q - q_ref||^2 + 1/2 ||p - p_ref||^2 + 1/4 sum((q - q_ref).^4)
        function v_field!(v, t, q, p, params)
            @inbounds for i in eachindex(v, p)
                v[i] = p[i] - p_ref[i]
            end
            nothing
        end
        function f_field!(f, t, q, p, params)
            @inbounds for i in eachindex(f, q)
                dq = q[i] - q_ref[i]
                f[i] = -dq - dq^3
            end
            nothing
        end
        function h_field(t, q, p, params)
            T(0.5) * (sum(abs2, q .- q_ref) + sum(abs2, p .- p_ref)) +
            T(0.25) * sum(x -> x^4, q .- q_ref)
        end

        ics_eq = (q = StateVariable(copy(q_ref)), p = StateVariable(copy(p_ref)))

        rs_jac = HRedSys(
            full_dim, reduced_dim, encoder(nn), decoder(nn),
            v_field!, f_field!, h_field,
            (zero(T), T(1.0)), T(0.05), ics_eq;
            integrator = ImplicitMidpoint(),
            q_ref = q_ref,
            p_ref = p_ref,
            pullback_mode = :jacobian
        )
        rs_vjp = HRedSys(
            full_dim, reduced_dim, encoder(nn), decoder(nn),
            v_field!, f_field!, h_field,
            (zero(T), T(1.0)), T(0.05), ics_eq;
            integrator = ImplicitMidpoint(),
            q_ref = q_ref,
            p_ref = p_ref,
            pullback_mode = :vjp
        )

        # 1. Reference-state shifting (Proposition 13.1):
        # For WeightTiedSymplecticAutoencoder, Ψ^dec(0) = 0 and Ψ^enc(0) = 0, so with
        # reference shift (q_ref, p_ref), ξ = (0, 0) lifts identically to (q_ref, p_ref)
        # and (q_ref, p_ref) encodes identically to ξ = (0, 0).
        ξ_zero = zeros(T, reduced_dim)
        z_lifted_zero = decode(rs_jac, ξ_zero)
        @test eltype(z_lifted_zero) === T
        @test isapprox(
            z_lifted_zero, vcat(q_ref, p_ref); atol = 10 * eps(T), rtol = 10 * eps(T))
        qp_lifted_zero = decode(rs_jac, (q = zeros(T, n), p = zeros(T, n)))
        @test isapprox(qp_lifted_zero.q, q_ref; atol = 10 * eps(T), rtol = 10 * eps(T))
        @test isapprox(qp_lifted_zero.p, p_ref; atol = 10 * eps(T), rtol = 10 * eps(T))
        ξ_enc_ref = encode(rs_jac, vcat(q_ref, p_ref))
        @test eltype(ξ_enc_ref) === T
        @test norm(ξ_enc_ref) <= 10 * eps(T)
        qp_enc_ref = encode(rs_jac, (q = q_ref, p = p_ref))
        @test norm(qp_enc_ref.q) <= 10 * eps(T) && norm(qp_enc_ref.p) <= 10 * eps(T)

        # Perturbation δz around (q_ref, p_ref)
        δq = randn(rng, T, N)
        δp = randn(rng, T, N)
        ξ_pert = encode(rs_jac, vcat(q_ref .+ δq, p_ref .+ δp))
        @test isapprox(
            ξ_pert, encoder(nn)(vcat(δq, δp)); atol = 10 * eps(T), rtol = 10 * eps(T))
        qp_pert = encode(rs_jac, (q = q_ref .+ δq, p = p_ref .+ δp))
        qp_pert_exp = encoder(nn)((q = δq, p = δp))
        @test isapprox(qp_pert.q, qp_pert_exp.q; atol = 10 * eps(T), rtol = 10 * eps(T))
        @test isapprox(qp_pert.p, qp_pert_exp.p; atol = 10 * eps(T), rtol = 10 * eps(T))

        # 2. VJP vs Jacobian agreement to 50 eps(T) for 2N = 12, 2n = 4 in Float32 and Float64 (catches a sign error or a swapped block in the pullback)
        for trial in 1:5
            q̃ = randn(rng, T, n)
            p̃ = randn(rng, T, n)
            v_jac = zeros(T, n)
            f_jac = zeros(T, n)
            v_vjp = zeros(T, n)
            f_vjp = zeros(T, n)

            rs_jac.v_reduced(v_jac, zero(T), q̃, p̃, nothing)
            rs_jac.f_reduced(f_jac, zero(T), q̃, p̃, nothing)
            rs_vjp.v_reduced(v_vjp, zero(T), q̃, p̃, nothing)
            rs_vjp.f_reduced(f_vjp, zero(T), q̃, p̃, nothing)

            # Verify non-zero vector field so a sign flip or zeroing is caught
            @test norm(v_jac) > T(1e-2)
            @test norm(f_jac) > T(1e-2)
            @test norm(v_vjp - v_jac) <= 50 * eps(T) * max(one(T), norm(v_jac))
            @test norm(f_vjp - f_jac) <= 50 * eps(T) * max(one(T), norm(f_jac))
        end

        # Also test standalone build_v_reduced and build_f_reduced directly
        v_red_j = build_v_reduced(
            v_field!, f_field!, decoder(nn);
            pullback_mode = :jacobian, q_ref = q_ref, p_ref = p_ref)
        f_red_j = build_f_reduced(
            v_field!, f_field!, decoder(nn);
            pullback_mode = :jacobian, q_ref = q_ref, p_ref = p_ref)
        v_red_v = build_v_reduced(
            v_field!, f_field!, decoder(nn);
            pullback_mode = :vjp, q_ref = q_ref, p_ref = p_ref)
        f_red_v = build_f_reduced(
            v_field!, f_field!, decoder(nn);
            pullback_mode = :vjp, q_ref = q_ref, p_ref = p_ref)

        q̃_test = randn(rng, T, n)
        p̃_test = randn(rng, T, n)
        vj_out = zeros(T, n)
        fj_out = zeros(T, n)
        vv_out = zeros(T, n)
        fv_out = zeros(T, n)
        v_red_j(vj_out, zero(T), q̃_test, p̃_test, nothing)
        f_red_j(fj_out, zero(T), q̃_test, p̃_test, nothing)
        v_red_v(vv_out, zero(T), q̃_test, p̃_test, nothing)
        f_red_v(fv_out, zero(T), q̃_test, p̃_test, nothing)
        @test norm(vv_out - vj_out) <= 50 * eps(T) * max(one(T), norm(vj_out))
        @test norm(fv_out - fj_out) <= 50 * eps(T) * max(one(T), norm(fj_out))

        # 3. Preallocated work buffers & ForwardDiff.Dual cache in HRedSys:
        @test hasfield(typeof(rs_jac), :ξ_buf)
        @test hasfield(typeof(rs_jac), :v_intermediate)
        @test hasfield(typeof(rs_jac), :f_intermediate)
        @test hasfield(typeof(rs_jac), :jac_result)
        @test hasfield(typeof(rs_jac), :dual_cache)
        @test rs_jac.dual_cache isa IdDict{DataType, Any}

        # Evaluate with ForwardDiff.Dual to populate dual_cache and verify agreement
        J_v_dual = ForwardDiff.jacobian(
            q -> begin
                out = similar(q, n)
                rs_jac.v_reduced(out, zero(T), q, p̃_test, nothing)
                out
            end,
            q̃_test
        )
        @test size(J_v_dual) == (n, n) && all(isfinite, J_v_dual)
        @test !isempty(rs_jac.dual_cache)

        # The Jacobian of the reduced field ξ = (q̃, p̃) ↦ (v, f) through the Dual path, with the
        # non-zero reference shift (q_ref, p_ref) above, equals the central finite differences of
        # that field. The finite differences run on Float64 states only, which the Float32 cache of
        # `rs_jac` does not take; they use `h = cbrt(eps)`, the step that balances the truncation
        # error `O(h^2)` (measured: about 0.2 h^2 relative at h = 1e-4) and the round-off error
        # `O(eps / h)` (about 4e-11 relative). The mismatch is relative to `max(1, ‖J‖)`, and
        # `1e-9` is 25 times the largest of 5 mismatches that were measured with this setup (3.8e-11),
        # and 9 orders below the mismatch of a transposed or a sign-flipped Jacobian (at least 1.6).
        if T === Float64
            rng_fd = Random.MersenneTwister(505)
            function reduced_field(ξ)
                out_v = zeros(eltype(ξ), n)
                out_f = zeros(eltype(ξ), n)
                rs_jac.v_reduced(
                    out_v, zero(eltype(ξ)), ξ[1:n], ξ[(n + 1):(2 * n)], nothing)
                rs_jac.f_reduced(
                    out_f, zero(eltype(ξ)), ξ[1:n], ξ[(n + 1):(2 * n)], nothing)
                vcat(out_v, out_f)
            end
            h_fd = cbrt(eps(T))
            for trial in 1:5
                ξ_fd = randn(rng_fd, T, reduced_dim)
                J_dual = ForwardDiff.jacobian(reduced_field, ξ_fd)
                J_fd = similar(J_dual)
                for j in 1:reduced_dim
                    e_j = zeros(T, reduced_dim)
                    e_j[j] = h_fd
                    J_fd[:, j] = (reduced_field(ξ_fd + e_j) - reduced_field(ξ_fd - e_j)) /
                                 (2 * h_fd)
                end
                mismatch(J) = norm(J - J_fd) / max(one(T), norm(J_fd))
                @test size(J_dual) == (reduced_dim, reduced_dim)
                @test mismatch(J_dual) <= 1e-9
                # The comparison can fail: a transposed and a sign-flipped Jacobian are rejected.
                @test mismatch(transpose(J_dual)) > 1e-3
                @test mismatch(-J_dual) > 1e-3
            end
        end

        # A state of another floating-point type than the default cache's (the decoder's
        # element type) raises an ArgumentError that names both types.
        S = T === Float32 ? Float64 : Float32
        msg_cache = "got a state of element type $S, but its cache has element type $T."
        @test_throws msg_cache v_red_j(
            zeros(S, n), zero(S), S.(q̃_test), S.(p̃_test), nothing)
        @test_throws msg_cache f_red_v(
            zeros(S, n), zero(S), S.(q̃_test), S.(p̃_test), nothing)

        # No result holds memory of the cache: mutating a result does not change the next
        # call, and a later call does not change an earlier result.
        cache_a = ReducedVFCache(T, N, n)
        q̃_other, p̃_other = randn(rng, T, n), randn(rng, T, n)
        for helper in (evaluate_vf_and_compute_∇Ψ, evaluate_vf_and_vjp)
            call(q, p) = helper(zero(T), q, p, nothing, decoder(nn), v_field!, f_field!,
                cache_a, q_ref, p_ref)
            r1 = call(q̃_test, p̃_test)
            r1_saved = deepcopy(r1)
            foreach(x -> fill!(x, T(NaN)), r1 isa Tuple ? r1 : (r1,))
            r2 = call(q̃_test, p̃_test)
            @test all(map((a, b) -> isapprox(a, b; rtol = 8 * eps(T)),
                r2 isa Tuple ? r2 : (r2,), r1_saved isa Tuple ? r1_saved : (r1_saved,)))
            call(q̃_other, p̃_other)
            @test all(x -> all(isnan, x), r1 isa Tuple ? r1 : (r1,))
        end

        # 4. Quadratic Hamiltonian H(q, p) = 1/2 ||(q - q_ref)^2 + (p - p_ref)^2||
        # with equilibrium (q_ref, p_ref) and integrator = ImplicitMidpoint():
        # integrate_reduced_system starting at ics = (q = q_ref, p = p_ref) keeps
        # ξ(t) = (0, 0) and (q(t), p(t)) = (q_ref, p_ref) to 10 eps(T) for 20 steps (catches a lift or an encoding that ignores the reference shift).
        function v_quad!(v, t, q, p, params)
            @inbounds for i in eachindex(v, p)
                v[i] = p[i] - p_ref[i]
            end
            nothing
        end
        function f_quad!(f, t, q, p, params)
            @inbounds for i in eachindex(f, q)
                f[i] = -(q[i] - q_ref[i])
            end
            nothing
        end
        function h_quad(t, q, p, params)
            T(0.5) * (sum(abs2, q .- q_ref) + sum(abs2, p .- p_ref))
        end

        rs_eq = HRedSys(
            full_dim, reduced_dim, encoder(nn), decoder(nn),
            v_quad!, f_quad!, h_quad,
            (zero(T), T(1.0)), T(0.05), # 20 steps
            (q = StateVariable(copy(q_ref)), p = StateVariable(copy(p_ref)));
            integrator = ImplicitMidpoint(),
            q_ref = q_ref,
            p_ref = p_ref
        )
        sol_red = integrate_reduced_system(rs_eq)
        @test length(sol_red.q) == 21 # initial state + 20 steps
        for step_idx in 0:20
            q̃_k = Vector(sol_red.q[step_idx])
            p̃_k = Vector(sol_red.p[step_idx])
            @test norm(q̃_k) <= 10 * eps(T)
            @test norm(p̃_k) <= 10 * eps(T)
            lifted_k = decode(rs_eq, (q = q̃_k, p = p̃_k))
            @test norm(lifted_k.q - q_ref) <= 10 * eps(T) * max(one(T), norm(q_ref))
            @test norm(lifted_k.p - p_ref) <= 10 * eps(T) * max(one(T), norm(p_ref))
        end

        # Non-equilibrium trajectory check: verify iterate moves when ics != (q_ref, p_ref)
        q_init = q_ref .+ T(0.2) .* randn(rng, T, N)
        p_init = p_ref .+ T(0.2) .* randn(rng, T, N)
        rs_neq = HRedSys(
            full_dim, reduced_dim, encoder(nn), decoder(nn),
            v_quad!, f_quad!, h_quad,
            (zero(T), T(0.2)), T(0.05),
            (q = StateVariable(q_init), p = StateVariable(p_init));
            integrator = ImplicitMidpoint(),
            q_ref = q_ref,
            p_ref = p_ref,
            pullback_mode = :jacobian
        )
        sol_red_neq = integrate_reduced_system(rs_neq)
        @test norm(Vector(sol_red_neq.q[1]) - Vector(sol_red_neq.q[0])) > T(1e-4)
        lifted_0 = decode(
            rs_neq, (
                q = Vector(sol_red_neq.q[0]), p = Vector(sol_red_neq.p[0]))
        )
        lifted_1 = decode(
            rs_neq, (
                q = Vector(sol_red_neq.q[1]), p = Vector(sol_red_neq.p[1]))
        )
        @test norm(lifted_1.q - lifted_0.q) > T(1e-4)

        # Verify build_h_reduced evaluates decoder((q = q̃, p = p̃)) with and without reference shift
        q̃_test = Vector(sol_red_neq.q[1])
        p̃_test = Vector(sol_red_neq.p[1])
        @test isapprox(
            rs_neq.h_reduced(zero(T), q̃_test, p̃_test, nothing),
            h_quad(zero(T), lifted_1.q, lifted_1.p, nothing);
            atol = 10 * eps(T), rtol = 10 * eps(T)
        )
        rs_unshifted = HRedSys(
            full_dim, reduced_dim, encoder(nn), decoder(nn),
            v_quad!, f_quad!, h_quad,
            (zero(T), T(0.2)), T(0.05),
            (q = StateVariable(q_init), p = StateVariable(p_init));
            integrator = ImplicitMidpoint(),
            pullback_mode = :jacobian
        )
        dec_unshifted = decoder(nn)((q = q̃_test, p = p̃_test))
        @test isapprox(
            rs_unshifted.h_reduced(zero(T), q̃_test, p̃_test, nothing),
            h_quad(zero(T), dec_unshifted.q, dec_unshifted.p, nothing);
            atol = 10 * eps(T), rtol = 10 * eps(T)
        )

        if T === Float32
            q_ref_f64 = Float64.(q_ref)
            p_ref_f64 = Float64.(p_ref)
            h_quad_generic(t, q, p, params) = eltype(q)(0.5) *
                                              (sum(abs2, q .- rs_neq.q_ref) +
                                               sum(abs2, p .- rs_neq.p_ref))
            rs_mixed_ref = HRedSys(
                full_dim, reduced_dim, encoder(nn), decoder(nn),
                v_quad!, f_quad!, h_quad_generic,
                (zero(T), T(0.2)), T(0.05),
                (q = StateVariable(q_init), p = StateVariable(p_init));
                integrator = ImplicitMidpoint(),
                q_ref = q_ref_f64,
                p_ref = p_ref_f64,
                pullback_mode = :vjp
            )
            @test eltype(rs_mixed_ref.q_ref) === Float32
            @test eltype(rs_mixed_ref.p_ref) === Float32
            h_val = @inferred rs_mixed_ref.h_reduced(
                zero(Float32), q̃_test, p̃_test, nothing)
            @test h_val isa Float32
            @test isapprox(
                h_val,
                rs_neq.h_reduced(zero(Float32), q̃_test, p̃_test, nothing);
                atol = 10 * eps(Float32), rtol = 10 * eps(Float32)
            )
            v_mixed = zeros(Float32, n)
            f_mixed = zeros(Float32, n)
            rs_mixed_ref.v_reduced(v_mixed, zero(Float32), q̃_test, p̃_test, nothing)
            rs_mixed_ref.f_reduced(f_mixed, zero(Float32), q̃_test, p̃_test, nothing)
            v_expected = zeros(Float32, n)
            f_expected = zeros(Float32, n)
            rs_neq.v_reduced(v_expected, zero(Float32), q̃_test, p̃_test, nothing)
            rs_neq.f_reduced(f_expected, zero(Float32), q̃_test, p̃_test, nothing)
            @test isapprox(
                v_mixed, v_expected; atol = 50 * eps(Float32), rtol = 50 * eps(Float32))
            @test isapprox(
                f_mixed, f_expected; atol = 50 * eps(Float32), rtol = 50 * eps(Float32))
        end
    end
end

@testset "reduction_error and projection_error with a non-zero reference shift ($T)" for T in REAL_ELTYPES
    rng = Random.MersenneTwister(606)
    N = 6
    n = 2
    arch = WeightTiedSymplecticAutoencoder(
        2 * N, 2 * n; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)
    nn = NeuralNetwork(arch, CPU(), T; rng = rng)
    q_ref = randn(rng, T, N)
    p_ref = randn(rng, T, N)
    z_ref = vcat(q_ref, p_ref)

    v_quad!(v, t, q, p, params) = (v .= p .- p_ref; nothing)
    f_quad!(f, t, q, p, params) = (f .= .-(q .- q_ref); nothing)
    h_quad(t, q, p, params) = T(0.5) * (sum(abs2, q .- q_ref) + sum(abs2, p .- p_ref))

    ics = (q = StateVariable(q_ref .+ T(0.3) .* randn(rng, T, N)),
        p = StateVariable(p_ref .+ T(0.3) .* randn(rng, T, N)))
    rs = HRedSys(
        2 * N, 2 * n, encoder(nn), decoder(nn), v_quad!, f_quad!, h_quad,
        (zero(T), T(0.25)), T(0.05), ics;
        integrator = ImplicitMidpoint(), q_ref = q_ref, p_ref = p_ref)
    sol_full = integrate_full_system(rs)
    sol_red = integrate_reduced_system(rs)

    # 5 steps store 6 states, as many as N = 6: a shift broadcast along the time axis instead of
    # the axis of the degrees of freedom runs without a dimension error, so only the value shows it.
    @test length(sol_full.q) == N
    @test length(sol_red.q) == N

    # The expected errors come from a loop over the stored states, with the shift added to and
    # subtracted from each state as a vector, not from the broadcasts of `decode` and `encode`.
    # `_norm` of a `(q, p)` pair is `(‖q‖ + ‖p‖) / √2`, and the `√2` cancels in the ratio.
    function relative_error(reconstruct, shift)
        dq, dp, nq, np = zero(T), zero(T), zero(T), zero(T)
        for k in 0:(N - 1)
            q_k = Vector(sol_full.q[k])
            p_k = Vector(sol_full.p[k])
            ẑ = reconstruct(k, shift)
            dq += sum(abs2, ẑ[1:N] .- q_k)
            dp += sum(abs2, ẑ[(N + 1):(2 * N)] .- p_k)
            nq += sum(abs2, q_k)
            np += sum(abs2, p_k)
        end
        (sqrt(dq) + sqrt(dp)) / (sqrt(nq) + sqrt(np))
    end
    projected(k, shift) = decoder(nn)(encoder(nn)(
        vcat(Vector(sol_full.q[k]), Vector(sol_full.p[k])) .- shift)) .+ shift
    reduced(k, shift) = decoder(nn)(vcat(Vector(sol_red.q[k]), Vector(sol_red.p[k]))) .+
                        shift

    proj_expected = relative_error(projected, z_ref)
    red_expected = relative_error(reduced, z_ref)
    @test projection_error(rs, sol_full) isa T
    @test reduction_error(rs, sol_full, sol_red) isa T
    @test isapprox(projection_error(rs, sol_full), proj_expected; rtol = 50 * eps(T))
    @test isapprox(reduction_error(rs, sol_full, sol_red), red_expected; rtol = 50 * eps(T))
    # The one-argument forms integrate the systems themselves and agree.
    @test isapprox(projection_error(rs), proj_expected; rtol = 50 * eps(T))
    @test isapprox(reduction_error(rs), red_expected; rtol = 50 * eps(T))

    # The check can fail: the errors are not trivial, and the errors without the shift differ
    # from the shifted ones, so a dropped or misplaced shift changes the result.
    @test proj_expected > T(1e-2)
    @test red_expected > T(1e-2)
    @test abs(relative_error(projected, zero(z_ref)) - proj_expected) >
          T(1e-3) * proj_expected
    @test abs(relative_error(reduced, zero(z_ref)) - red_expected) > T(1e-3) * red_expected
end

@testset "build_v_reduced and build_f_reduced with one reference only ($mode, $which)" for mode in (:jacobian, :vjp),
    which in (:q_ref, :p_ref)

    rng = Random.MersenneTwister(707)
    T = Float64
    N = 6
    n = 2
    arch = WeightTiedSymplecticAutoencoder(
        2 * N, 2 * n; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)
    nn = NeuralNetwork(arch, CPU(), T; rng = rng)
    ref = randn(rng, T, N)
    # Both fields depend on q and on p, so a shift of the wrong component changes the result.
    v_nl!(v, t, q, p, params) = (v .= p .+ T(0.5) .* q .^ 2; nothing)
    f_nl!(f, t, q, p, params) = (f .= .-q .- p .^ 2; nothing)

    q̃, p̃ = randn(rng, T, n), randn(rng, T, n)
    ξ = vcat(q̃, p̃)
    # The expected values come from the decoder, its Jacobian from ForwardDiff, and a loop that
    # adds the reference to the component it belongs to.
    z = Vector(decoder(nn)(ξ))
    ∇Ψ = ForwardDiff.jacobian(x -> decoder(nn)(x), ξ)
    for i in 1:N
        which === :q_ref ? (z[i] += ref[i]) : (z[N + i] += ref[i])
    end
    v_z, f_z = zeros(T, N), zeros(T, N)
    v_nl!(v_z, zero(T), z[1:N], z[(N + 1):(2 * N)], nothing)
    f_nl!(f_z, zero(T), z[1:N], z[(N + 1):(2 * N)], nothing)
    v_expected = ∇Ψ[(N + 1):(2 * N), (n + 1):(2 * n)]' * v_z .-
                 ∇Ψ[1:N, (n + 1):(2 * n)]' * f_z
    f_expected = ∇Ψ[1:N, 1:n]' * f_z .- ∇Ψ[(N + 1):(2 * N), 1:n]' * v_z

    kw = (; which => ref)
    v_red = build_v_reduced(v_nl!, f_nl!, decoder(nn); pullback_mode = mode, kw...)
    f_red = build_f_reduced(v_nl!, f_nl!, decoder(nn); pullback_mode = mode, kw...)
    v_out, f_out = zeros(T, n), zeros(T, n)
    v_red(v_out, zero(T), q̃, p̃, nothing)
    f_red(f_out, zero(T), q̃, p̃, nothing)
    @test isapprox(v_out, v_expected; rtol = 1e-10, atol = 1e-12)
    @test isapprox(f_out, f_expected; rtol = 1e-10, atol = 1e-12)

    # The shift is not trivial: the unshifted fields differ from the shifted ones.
    v_red0 = build_v_reduced(v_nl!, f_nl!, decoder(nn); pullback_mode = mode)
    v_out0 = zeros(T, n)
    v_red0(v_out0, zero(T), q̃, p̃, nothing)
    @test norm(v_out0 .- v_expected) > 1e-3 * norm(v_expected)
end
