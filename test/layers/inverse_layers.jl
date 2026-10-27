using GeometricMachineLearning
using GeometricMachineLearning: ForwardDiff, params
using KernelAbstractions
using LinearAlgebra
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/poisson.jl")

Random.seed!(1234)

@testset "inverse_layer mapping and wrapper types" begin
    gl_q = GradientLayerQ(6, 12, tanh)
    gl_p = GradientLayerP(6, 12, tanh)
    igl_q = inverse_layer(gl_q)
    igl_p = inverse_layer(gl_p)

    @test igl_q isa InverseGradientLayerQ{6, 12, typeof(tanh)}
    @test igl_p isa InverseGradientLayerP{6, 12, typeof(tanh)}
    @test igl_q === InverseGradientLayerQ(6, 12, tanh)
    @test igl_p === InverseGradientLayerP(6, 12, tanh)
    @test GradientLayerQ{6, 12}(tanh) === gl_q
    @test GradientLayerP{6, 12}(tanh) === gl_p
    @test InverseGradientLayerQ{6, 12}(tanh) === igl_q
    @test InverseGradientLayerP{6, 12}(tanh) === igl_p
    @test inverse_layer(igl_q) === gl_q
    @test inverse_layer(igl_p) === gl_p
    @test inverse_layer(GradientLayerQ) === InverseGradientLayerQ
    @test inverse_layer(InverseGradientLayerQ) === GradientLayerQ
    @test inverse_layer(GradientLayerP) === InverseGradientLayerP
    @test inverse_layer(InverseGradientLayerP) === GradientLayerP

    sa_q = SymplecticAttentionQ(6, 3)
    sa_p = SymplecticAttentionP(6, 3)
    isa_q = inverse_layer(sa_q)
    isa_p = inverse_layer(sa_p)

    @test isa_q isa InverseSymplecticAttentionQ{6, 6}
    @test isa_p isa InverseSymplecticAttentionP{6, 6}
    @test isa_q === InverseSymplecticAttentionQ(6, 3)
    @test isa_p === InverseSymplecticAttentionP(6, 3)
    @test inverse_layer(isa_q) === sa_q
    @test inverse_layer(isa_p) === sa_p
    @test inverse_layer(SymplecticAttentionQ) === InverseSymplecticAttentionQ
    @test inverse_layer(InverseSymplecticAttentionQ) === SymplecticAttentionQ
    @test inverse_layer(SymplecticAttentionP) === InverseSymplecticAttentionP
    @test inverse_layer(InverseSymplecticAttentionP) === SymplecticAttentionP

    psd = PSDLayer{4, 10}()
    tpsd = inverse_layer(psd)
    @test tpsd === TransposePSDLayer{10, 4}()
    @test tpsd === TransposePSDLayer(10, 4)
    @test inverse_layer(tpsd) === psd
    @test inverse_layer(PSDLayer{4, 10}) === TransposePSDLayer{10, 4}
    @test inverse_layer(TransposePSDLayer{10, 4}) === PSDLayer{4, 10}
end

@testset "TransposePSDLayer input validation" begin
    r1 = r"^TransposePSDLayer dimensions \(3, 4\) must be positive even integers\.$"
    r2 = r"^TransposePSDLayer dimensions \(4, 5\) must be positive even integers\.$"
    r3 = r"^TransposePSDLayer dimensions \(0, 4\) must be positive even integers\.$"
    r4 = r"^TransposePSDLayer dimensions \(-2, 4\) must be positive even integers\.$"
    r5 = r"^TransposePSDLayer dimensions \(4, 0\) must be positive even integers\.$"
    r6 = r"^TransposePSDLayer dimensions \(:a, 4\) must be integers\.$"
    @test_throws r1 TransposePSDLayer(3, 4)
    @test_throws r2 TransposePSDLayer(4, 5)
    @test_throws r3 TransposePSDLayer(0, 4)
    @test_throws r4 TransposePSDLayer(-2, 4)
    @test_throws r5 TransposePSDLayer{4, 0}()
    @test_throws r6 TransposePSDLayer{:a, 4}()
end

@testset "exact layerwise inversion ($T)" for T in REAL_ELTYPES
    layers = (
        GradientLayerQ(6, 12, tanh),
        GradientLayerP(6, 12, tanh),
        SymplecticAttentionQ(6, 3),
        SymplecticAttentionP(6, 3),
        SymplecticAttentionQ(6; symmetric = false),
        SymplecticAttentionP(6; symmetric = false))
    zv = randn(T, 6)
    zm = randn(T, 6, 3)
    zt = randn(T, 6, 3, 2)
    nt_v = (q = zv[1:3], p = zv[4:6])
    nt_m = (q = zm[1:3, :], p = zm[4:6, :])
    nt_t = (q = zt[1:3, :, :], p = zt[4:6, :, :])

    for l in layers
        l_inv = inverse_layer(l)
        ps = params(NeuralNetwork(Chain(l), CPU(), T)).L1
        ps_inv = params(NeuralNetwork(Chain(l_inv), CPU(), T)).L1
        @test parameterlength(l_inv) == parameterlength(l)
        @test keys(ps_inv) == keys(ps)

        for z in (zv, zm, zt)
            z_fwd = l(z, ps)
            @test !(z_fwd ≈ z)
            @test norm(l_inv(z_fwd, ps) - z) <= 10 * eps(T) * max(one(T), norm(z))
            @test norm(l(l_inv(z, ps), ps) - z) <= 10 * eps(T) * max(one(T), norm(z))
        end

        for nt in (nt_v, nt_m, nt_t)
            nt_fwd = l(nt, ps)
            nt_rec1 = l_inv(nt_fwd, ps)
            nt_rec2 = l(l_inv(nt, ps), ps)
            @test norm(nt_rec1.q - nt.q) <= 10 * eps(T) * max(one(T), norm(nt.q))
            @test norm(nt_rec1.p - nt.p) <= 10 * eps(T) * max(one(T), norm(nt.p))
            @test norm(nt_rec2.q - nt.q) <= 10 * eps(T) * max(one(T), norm(nt.q))
            @test norm(nt_rec2.p - nt.p) <= 10 * eps(T) * max(one(T), norm(nt.p))
        end
    end

    # Check hand-built NamedTuple ps = (K = ..., b = ...) for GradientLayerQ/P and inverses
    K = randn(T, 6, 3)
    b = randn(T, 6)
    a = randn(T, 6)
    tol_v = 10 * eps(T) * max(one(T), norm(zv))
    gl_pairs = (
        (GradientLayerQ{6, 12}(tanh), InverseGradientLayerQ{6, 12}(tanh)),
        (GradientLayerP{6, 12}(tanh), InverseGradientLayerP{6, 12}(tanh)))
    for (gl, igl) in gl_pairs
        for ps_kb in ((K = K, b = b), (K = K, b = b, a = a), (weight = K, bias = b))
            @test norm(igl(gl(zv, ps_kb), ps_kb) - zv) <= tol_v
            @test norm(gl(igl(zv, ps_kb), ps_kb) - zv) <= tol_v
        end
    end
end

@testset "PSDLayer and TransposePSDLayer ($T)" for T in REAL_ELTYPES
    l = PSDLayer(4, 10)
    l_inv = inverse_layer(l)
    ps = params(NeuralNetwork(Chain(l), CPU(), T)).L1
    @test parameterlength(l_inv) == parameterlength(l)

    ξv = randn(T, 4)
    ξm = randn(T, 4, 3)
    ξt = randn(T, 4, 3, 2)
    for ξ in (ξv, ξm, ξt)
        z = l(ξ, ps)
        ξ_rec = l_inv(z, ps)
        @test norm(ξ_rec - ξ) <= 20 * eps(T) * max(one(T), norm(ξ))
    end

    nt_v = (q = ξv[1:2], p = ξv[3:4])
    nt_m = (q = ξm[1:2, :], p = ξm[3:4, :])
    nt_t = (q = ξt[1:2, :, :], p = ξt[3:4, :, :])
    for nt in (nt_v, nt_m, nt_t)
        nt_rec = l_inv(l(nt, ps), ps)
        @test norm(nt_rec.q - nt.q) <= 20 * eps(T) * max(one(T), norm(nt.q))
        @test norm(nt_rec.p - nt.p) <= 20 * eps(T) * max(one(T), norm(nt.p))
    end

    ∇l = ForwardDiff.jacobian(ξ -> l(ξ, ps), ξv)
    J10 = poisson_matrix(T, 5)
    J4 = poisson_matrix(T, 2)
    @test norm(transpose(∇l) * J10 * ∇l - J4) <= 20 * eps(T)

    # Square n == N case (catches O3 where PSDLayer{M, M} takes the else branch ps.weight')
    l_sq = PSDLayer(4, 4)
    l_sq_inv = inverse_layer(l_sq)
    θ = T(0.7)
    R = T[cos(θ) -sin(θ); sin(θ) cos(θ)]
    ps_sq = (weight = StiefelManifold(R),)
    @test norm(l_sq_inv(l_sq(ξv, ps_sq), ps_sq) - ξv) <= 20 * eps(T) * norm(ξv)
    @test norm(l_sq(l_sq_inv(ξv, ps_sq), ps_sq) - ξv) <= 20 * eps(T) * norm(ξv)

    # Downscaling PSDLayer(10, 4) and its inverse TransposePSDLayer(4, 10)
    l_down = PSDLayer(10, 4)
    l_down_inv = inverse_layer(l_down)
    @test norm(l_down(l_down_inv(ξv, ps), ps) - ξv) <= 20 * eps(T) * norm(ξv)
end
