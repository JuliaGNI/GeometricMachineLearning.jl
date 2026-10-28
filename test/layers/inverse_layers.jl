using GeometricMachineLearning
using GeometricMachineLearning: ForwardDiff, output_dimension, params
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

    @test igl_q isa InverseGradientLayerQ{6, 6, typeof(tanh)}
    @test igl_p isa InverseGradientLayerP{6, 6, typeof(tanh)}
    @test igl_q.second_dim == igl_p.second_dim == 12
    @test igl_q === InverseGradientLayerQ(6, 12, tanh)
    @test igl_p === InverseGradientLayerP(6, 12, tanh)
    @test inverse_layer(igl_q) === gl_q
    @test inverse_layer(igl_p) === gl_p
    @test hash(inverse_layer(igl_q)) == hash(gl_q)
    @test output_dimension(igl_q) == output_dimension(inverse_layer(igl_q)) == 6
    @test inverse_layer(GradientLayerQ) === InverseGradientLayerQ
    @test inverse_layer(InverseGradientLayerQ) === GradientLayerQ
    @test inverse_layer(GradientLayerP) === InverseGradientLayerP
    @test inverse_layer(InverseGradientLayerP) === GradientLayerP
    @test inverse_layer(GradientLayerQ{6, 6}) === InverseGradientLayerQ{6, 6}
    @test inverse_layer(InverseGradientLayerQ{6, 6}) === GradientLayerQ{6, 6}
    @test inverse_layer(GradientLayerP{6, 6}) === InverseGradientLayerP{6, 6}
    @test inverse_layer(InverseGradientLayerP{6, 6}) === GradientLayerP{6, 6}
    @test inverse_layer(typeof(gl_q)) === typeof(igl_q)
    @test inverse_layer(typeof(igl_p)) === typeof(gl_p)

    sa_q = SymplecticAttentionQ(6)
    sa_p = SymplecticAttentionP(6)
    isa_q = inverse_layer(sa_q)
    isa_p = inverse_layer(sa_p)

    @test isa_q isa InverseSymplecticAttentionQ{6, 6}
    @test isa_p isa InverseSymplecticAttentionP{6, 6}
    @test isa_q === InverseSymplecticAttentionQ(6)
    @test isa_p === InverseSymplecticAttentionP(6)
    @test inverse_layer(isa_q) === sa_q
    @test inverse_layer(isa_p) === sa_p
    @test inverse_layer(SymplecticAttentionQ) === InverseSymplecticAttentionQ
    @test inverse_layer(InverseSymplecticAttentionQ) === SymplecticAttentionQ
    @test inverse_layer(SymplecticAttentionP) === InverseSymplecticAttentionP
    @test inverse_layer(InverseSymplecticAttentionP) === SymplecticAttentionP
    @test inverse_layer(SymplecticAttentionQ{6, 6}) === InverseSymplecticAttentionQ{6, 6}
    @test inverse_layer(InverseSymplecticAttentionQ{6, 6}) === SymplecticAttentionQ{6, 6}
    @test inverse_layer(SymplecticAttentionP{6, 6}) === InverseSymplecticAttentionP{6, 6}
    @test inverse_layer(InverseSymplecticAttentionP{6, 6}) === SymplecticAttentionP{6, 6}

    psd = PSDLayer{4, 10}()
    tpsd = inverse_layer(psd)
    @test tpsd === TransposePSDLayer{10, 4}()
    @test tpsd === TransposePSDLayer(10, 4)
    @test inverse_layer(tpsd) === psd
    @test inverse_layer(PSDLayer{4, 10}) === TransposePSDLayer{10, 4}
    @test inverse_layer(TransposePSDLayer{10, 4}) === PSDLayer{4, 10}
end

@testset "TransposePSDLayer input validation" begin
    r1 = r"^(?:ArgumentError: )?TransposePSDLayer dimensions \(3, 4\) must be positive even integers\.$"
    r2 = r"^(?:ArgumentError: )?TransposePSDLayer dimensions \(4, 5\) must be positive even integers\.$"
    r3 = r"^(?:ArgumentError: )?TransposePSDLayer dimensions \(0, 4\) must be positive even integers\.$"
    r4 = r"^(?:ArgumentError: )?TransposePSDLayer dimensions \(-2, 4\) must be positive even integers\.$"
    r5 = r"^(?:ArgumentError: )?TransposePSDLayer dimensions \(4, 0\) must be positive even integers\.$"
    r6 = r"^(?:ArgumentError: )?TransposePSDLayer dimensions \(:?a, 4\) must be integers\.$"
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
        SymplecticAttentionQ(6),
        SymplecticAttentionP(6),
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
            z_fwd = @inferred l(z, ps)
            z_rec1 = @inferred l_inv(z_fwd, ps)
            z_rec2 = @inferred l(l_inv(z, ps), ps)
            @test eltype(z_fwd) === T
            @test eltype(z_rec1) === T
            @test eltype(z_rec2) === T
            @test !(z_fwd ≈ z)
            @test norm(z_rec1 - z) <= 10 * eps(T) * max(one(T), norm(z))
            @test norm(z_rec2 - z) <= 10 * eps(T) * max(one(T), norm(z))
        end

        for nt in (nt_v, nt_m, nt_t)
            nt_fwd = @inferred l(nt, ps)
            nt_rec1 = @inferred l_inv(nt_fwd, ps)
            nt_rec2 = @inferred l(l_inv(nt, ps), ps)
            @test eltype(nt_fwd.q) === T && eltype(nt_fwd.p) === T
            @test eltype(nt_rec1.q) === T && eltype(nt_rec1.p) === T
            @test eltype(nt_rec2.q) === T && eltype(nt_rec2.p) === T
            @test norm(nt_rec1.q - nt.q) <= 10 * eps(T) * max(one(T), norm(nt.q))
            @test norm(nt_rec1.p - nt.p) <= 10 * eps(T) * max(one(T), norm(nt.p))
            @test norm(nt_rec2.q - nt.q) <= 10 * eps(T) * max(one(T), norm(nt.q))
            @test norm(nt_rec2.p - nt.p) <= 10 * eps(T) * max(one(T), norm(nt.p))
        end
    end
end

@testset "PSDLayer and TransposePSDLayer ($T)" for T in REAL_ELTYPES
    l = PSDLayer(4, 10)
    l_inv = inverse_layer(l)
    ps = params(NeuralNetwork(Chain(l), CPU(), T)).L1
    @test parameterlength(l_inv) == parameterlength(l)
    @test eltype(ps.weight) === T

    ξv = randn(T, 4)
    ξm = randn(T, 4, 3)
    ξt = randn(T, 4, 3, 2)
    for ξ in (ξv, ξm, ξt)
        z = @inferred l(ξ, ps)
        ξ_rec = @inferred l_inv(z, ps)
        @test eltype(z) === T
        @test eltype(ξ_rec) === T
        @test norm(ξ_rec - ξ) <= 20 * eps(T) * max(one(T), norm(ξ))
    end

    nt_v = (q = ξv[1:2], p = ξv[3:4])
    nt_m = (q = ξm[1:2, :], p = ξm[3:4, :])
    nt_t = (q = ξt[1:2, :, :], p = ξt[3:4, :, :])
    for nt in (nt_v, nt_m, nt_t)
        nt_fwd = @inferred l(nt, ps)
        nt_rec = @inferred l_inv(nt_fwd, ps)
        @test eltype(nt_fwd.q) === T && eltype(nt_fwd.p) === T
        @test eltype(nt_rec.q) === T && eltype(nt_rec.p) === T
        @test norm(nt_rec.q - nt.q) <= 20 * eps(T) * max(one(T), norm(nt.q))
        @test norm(nt_rec.p - nt.p) <= 20 * eps(T) * max(one(T), norm(nt.p))
    end

    ∇l = ForwardDiff.jacobian(ξ -> l(ξ, ps), ξv)
    J10 = poisson_matrix(10, T)
    J4 = poisson_matrix(4, T)
    @test norm(transpose(∇l) * J10 * ∇l - J4) <= 20 * eps(T)

    # Square n == N case where PSDLayer{M, M} takes the else branch ps.weight'
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

@testset "Error cases" begin
    @test GeometricMachineLearning.compute_iterations_for_symplectic_system(10, 4, 1) ==
          [4, 10]
    @test GeometricMachineLearning.compute_iterations_for_symplectic_system(6, 6, 2) ==
          [6, 6, 6]
    @test_throws ArgumentError WeightTiedSymplecticAutoencoder(0, 4)
    @test_throws ArgumentError WeightTiedSymplecticAutoencoder(10, 4; n_blocks = 0)
    @test WeightTiedSymplecticAutoencoder(4, 4; n_layers = 0) isa
          WeightTiedSymplecticAutoencoder{:InitQ}
    @test WeightTiedSymplecticAutoencoder(4, 2; init_q = false) isa
          WeightTiedSymplecticAutoencoder{:InitP}
    @test GeometricMachineLearning._reverse_tuple((1, 2, 3)) === (3, 2, 1)

    igl = InverseGradientLayerQ(6, 12, tanh)
    ps_gl = (weight = randn(6, 3), bias = randn(6), scale = randn(6))
    @test_throws Exception igl(randn(7, 3), ps_gl)
    @test_throws Exception igl((q = randn(4, 3), p = randn(3, 3)), ps_gl)
    @test_throws AssertionError InverseGradientLayerQ(5, 12, tanh)
    @test_throws AssertionError InverseGradientLayerP(6, 11, tanh)
    @test inverse_layer(InverseGradientLayerQ(6, 12, tanh)) != GradientLayerQ(6, 8, tanh)
    @test_throws MethodError SymplecticAttentionQ(6, 3)
    @test_throws MethodError InverseSymplecticAttentionQ(6, 3)

    isa_l = InverseSymplecticAttentionQ(6; symmetric = false)
    ps_sa = (A = randn(3, 3),)
    @test_throws Exception isa_l(randn(7, 3), ps_sa)
    @test_throws Exception isa_l((q = randn(4, 3), p = randn(3, 3)), ps_sa)

    tpsd = TransposePSDLayer(10, 4)
    ps_psd = params(NeuralNetwork(Chain(tpsd), CPU(), Float64)).L1
    @test_throws Exception tpsd(randn(7, 3), ps_psd)
    @test_throws Exception tpsd((q = randn(4, 3), p = randn(5, 3)), ps_psd)
end
