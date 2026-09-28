using GeometricMachineLearning
using PoincareInvariants: CanonicalFirstPI, compute!
using Test
import Random

Random.seed!(123)

# `symplectic_autoencoder_tests.jl` and `psd_architecture_tests.jl` check symplecticity of a decoder
# pointwise: ``(\nabla_z\Psi)^T\mathbb{J}_{2N}\nabla_z\Psi = \mathbb{J}_{2n}`` at a single random
# latent point. This file checks the integrated consequence of that identity: for a symplectic
# ``\Psi`` and any contractible closed loop ``\gamma`` in latent space,
#
#     ``\oint_{\Psi\circ\gamma}p\cdot{}dq = \oint_\gamma{}p\cdot{}dq,``
#
# the first Poincaré integral invariant (Arnold, *Mathematical Methods of Classical Mechanics*,
# §44; Hairer, Lubich and Wanner, *Geometric Numerical Integration*, VI.2).
#
# This is the *weaker* statement of the two, and it is here beside the Jacobian test rather than in
# place of it. The pointwise condition implies it -- if ``\Psi^*\Omega = \Omega`` everywhere then
# Stokes turns the two loop integrals into the same surface integral -- while the converse fails: a
# loop integral collapses a matrix identity to one scalar, so violations of opposite sign along the
# loop cancel. What it adds is coverage the Jacobian test cannot give. It looks at a finite loop
# rather than one point and a first-order expansion around it, and it costs `K` forward decoder
# evaluations rather than a materialised ``2N\times{}2n`` Jacobian.
#
# The quadrature is `PoincareInvariants.jl`'s. `CanonicalFirstPI` reads ``(q, p)`` in the same
# stacking this package uses, so its ``\oint\theta\cdot{}dz`` is ``\oint{}p\cdot{}dq``, sign
# convention included. Its default `FirstFourierPlan` differentiates the loop spectrally, and that
# matters here: the obvious alternative, a shoelace sum over the polygon through the sampled points,
# is only second order however smooth the curve is, and at the sample counts used below its
# discretisation error alone would be larger than every defect these tests are trying to bound.

raw"""
    loop_invariant(Z)

First Poincaré integral invariant ``\oint{}p\cdot{}dq`` of the closed loop whose points are the
columns of `Z`, a ``2d\times{}K`` matrix in ``(q, p)`` stacking.

The loop must be closed, traversed exactly once, with the last column *not* repeating the first.
`compute!` wants the points as rows and accepts a lazy adjoint, so `Z'` costs no copy.
"""
function loop_invariant(Z::AbstractMatrix{T}) where {T}
    D, K = size(Z)
    compute!(CanonicalFirstPI{T, D}(K), Z')
end

raw"""
    symplecticity_defect(Ψ, Z)

Discrepancy between ``\oint{}p\cdot{}dq`` before and after applying `Ψ` to the loop `Z`, relative to
the invariant of `Z`. Zero for a symplectic `Ψ`, up to the resolution of the loop and the working
precision of `Ψ`.

`Ψ` is applied to the whole ``2d\times{}K`` loop in one batched call.
"""
function symplecticity_defect(Ψ, Z)
    before = loop_invariant(Z)
    after = loop_invariant(Ψ(Z))
    abs(after - before) / abs(before)
end

raw"""
    test_loop(D, K; T, radius, closed_fraction)

A smooth loop in ``\mathbb{R}^D`` sampled at `K` points, as a ``D\times{}K`` matrix in ``(q, p)``
stacking.

Each conjugate pair ``(q_i, p_i)`` traces a circle of radius `radius`, at one of three frequencies
and with its own phase, so the loop is closed and periodic in the sample index -- what the spectral
quadrature needs. The two members of a pair must share a frequency: give them different ones and
every pair contributes zero to ``\oint{}p\cdot{}dq``, leaving an invariant of zero to divide a
relative defect by. `radius` scales the loop, and it has to: a decoder is close to linear near the
origin, so only a large enough loop exercises its nonlinearity. `closed_fraction` below `1` stops
the loop short of its period, which is how the closure precondition gets violated on purpose.
"""
function test_loop(D::Integer, K::Integer;
        T::Type = Float64, radius::Real = 1, closed_fraction::Real = 1)
    # A closed loop drops the endpoint, which would repeat the first point; an open one keeps it.
    s = closed_fraction == 1 ? T.(range(0, 1, length = K + 1))[1:K] :
        T.(range(0, closed_fraction, length = K))
    d = D ÷ 2
    Z = Matrix{T}(undef, D, K)
    for i in 1:d
        θ = 2π .* (mod(i - 1, 3) + 1) .* s .+ T(0.37i)
        Z[i, :] .= T(radius) .* cos.(θ)
        Z[d + i, :] .= T(radius) .* sin.(θ)
    end
    Z
end

# The floor these tests sit on is the network's working precision, not the quadrature's: measured
# over the loops below it is around `1e-16` in `Float64` and `2e-7` in `Float32`, so `sqrt(eps(T))`
# leaves room in both and is still far below the defect of a map that is not symplectic.
tolerance(T::Type) = sqrt(eps(T))

@testset "Analytic value and sign convention" begin
    # The unit circle traversed counterclockwise encloses area ``\pi``, and ``\oint{}p\cdot{}dq``
    # is the *negative* of the enclosed area for that orientation. This pins the quadrature and
    # the sign at the same time; the integral is signed, so the two sides of a comparison have to
    # be traversed alike.
    for K in (64, 512)
        s = range(0, 1, length = K + 1)[1:K]
        counterclockwise = permutedims(hcat(cos.(2π .* s), sin.(2π .* s)))
        @test loop_invariant(counterclockwise) ≈ -π
        @test loop_invariant(counterclockwise[:, [1; K:-1:2]]) ≈ π
    end
end

@testset "Linear symplectic decoder ($N, $n)" for (N, n) in ((10, 6), (20, 10))
    psd_nn = NeuralNetwork(PSDArch(N, n))
    Z = test_loop(n, 512)

    @test symplecticity_defect(decoder(psd_nn), Z) < tolerance(Float64)

    # And after the decomposition has actually been computed.
    solve!(psd_nn, DataLoader(rand(N, 10 * N); autoencoder = true))
    @test symplecticity_defect(decoder(psd_nn), Z) < tolerance(Float64)
end

@testset "SymplecticAutoencoder decoder ($T, $N, $n)" for T in (Float32, Float64),
    (N, n) in ((10, 6), (20, 10))

    sae_nn = NeuralNetwork(SymplecticAutoencoder(N, n), CPU(), T)
    Z = test_loop(n, 512; T = T, radius = 3)

    # Symplecticity is architectural, so it holds before any training at all.
    @test symplecticity_defect(decoder(sae_nn), Z) < tolerance(T)

    # And it survives a short training run, as the pointwise test also asserts.
    dl = DataLoader(rand(T, N, 10 * N); autoencoder = true)
    Optimizer(Adam(T), sae_nn)(sae_nn, dl, Batch(10), 10; show_progress = false)
    @test symplecticity_defect(decoder(sae_nn), Z) < tolerance(T)
end

@testset "A decoder that is not symplectic fails the test" begin
    # Without this the suite would pass vacuously: a diagnostic that every map satisfies says
    # nothing about the ones that are supposed to. The shape is a decoder's -- the PSD lift up to
    # the full dimension -- followed by a ResNet, which has no reason to preserve anything.
    N, n = 10, 6
    not_symplectic = NeuralNetwork(Chain(PSDLayer(n, N), Chain(ResNet(N, 2, tanh)).layers...))

    # Percent-level and up, against the `1e-16` an exactly symplectic decoder returns.
    for radius in (0.5, 1.0, 2.0)
        @test symplecticity_defect(not_symplectic, test_loop(n, 512; radius = radius)) > 1e-3
    end
end

@testset "Resolution of the loop" begin
    # The identity holds for the curve, but the number is computed from samples of it, and the
    # resolution needed is set by the curvature of the *image* -- a latent loop that looks amply
    # sampled can still be badly under-resolved once the decoder has bent it. Under-resolution then
    # reads as a loss of symplecticity, which is the trap this diagnostic sets for its user.
    N, n = 10, 6
    sae_decoder = decoder(NeuralNetwork(SymplecticAutoencoder(N, n)))
    # Large enough that the decoder is strongly nonlinear along it, so the failure mode is
    # reachable at all: on a gentle loop every K passes and the test asserts nothing.
    defect_at(K) = symplecticity_defect(sae_decoder, test_loop(n, K; radius = 25))

    # At a sufficient K the defect is at the floor and stays there when K is doubled.
    @test defect_at(1024) < tolerance(Float64)
    @test defect_at(2048) < tolerance(Float64)

    # Deliberately under-resolved, the same exactly symplectic decoder reports a defect orders of
    # magnitude above that floor.
    @test defect_at(32) > 100 * tolerance(Float64)
end

@testset "A loop that is not closed" begin
    # Closure is the sharp precondition. An open loop leaves ``\oint{}p\cdot{}dq`` undefined, and
    # the defect it produces does not converge as the loop is refined -- refinement resolves the
    # curve, it does not close it -- so the number can be mistaken for a property of the decoder
    # at any `K`.
    N, n = 10, 6
    sae_decoder = decoder(NeuralNetwork(SymplecticAutoencoder(N, n)))
    open_defect(K) = symplecticity_defect(sae_decoder,
        test_loop(n, K; radius = 3, closed_fraction = 0.9))

    @test open_defect(256) > tolerance(Float64)
    @test open_defect(1024) > tolerance(Float64)
    @test open_defect(4096) > tolerance(Float64)
end
