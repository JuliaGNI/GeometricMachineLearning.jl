using GeometricMachineLearning
using GeometricMachineLearning: HRedSys, build_v_reduced, build_f_reduced,
                                integrate_reduced_system, evaluate_vf_and_compute_∇Ψ,
                                ForwardDiff, StateVariable
using GeometricIntegrators: ImplicitMidpoint
using KernelAbstractions
using LinearAlgebra
using Test
using Random: Random

include("../helpers/eltypes.jl")

Random.seed!(42)

@testset "Clause 3: HRedSys reference-state shifting, pullback_mode = :jacobian | :vjp, and work buffers" begin
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

        # 2. VJP vs Jacobian agreement to 50 eps(T) for 2N = 12, 2n = 4 in Float32 and Float64 (catches M2!)
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

            # Verify non-zero vector field so a sign flip (M2) or zeroing is caught
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

        # 4. Quadratic Hamiltonian H(q, p) = 1/2 ||(q - q_ref)^2 + (p - p_ref)^2||
        # with equilibrium (q_ref, p_ref) and integrator = ImplicitMidpoint():
        # integrate_reduced_system starting at ics = (q = q_ref, p = p_ref) keeps
        # ξ(t) = (0, 0) and (q(t), p(t)) = (q_ref, p_ref) to 10 eps(T) for 20 steps (catches M3!).
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
    end
end
