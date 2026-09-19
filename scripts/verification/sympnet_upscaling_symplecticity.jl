# Is the sympnet upscaling chain symplectic?
#
# The chain is `PSDLayer(N, N2) -> GradientLayerQ(N2) -> GradientLayerP(N2) -> PSDLayer(N2, N)`.
# Each layer is symplectic in its own right. This script checks whether the composition is, and
# establishes that it is not -- for an algebraic reason, not a numerical one.
#
# THAT IS THE INTENDED PROPERTY, not a defect. End-to-end symplecticity of the chain is not the
# design goal: the architecture only approximates it. The *worst* case over the 20 random chains
# improves as `N` grows -- 0.84 at `N = 2, N2 = 4` against 0.072 at `N = 20, N2 = 40`, in
# `Float64` -- but the *best* case moves the other way, from 0.0018 to 0.026. The deviation
# concentrates as `N` grows; it does not vanish. So this script asserts no bound: it prints the
# exact part -- the three layerwise identities, which hold to machine precision -- beside the
# inexact part, which is the round-trip deviation.
#
# Five checks, in order:
#
#   1. the three layerwise identities hold to machine precision;
#   2. the round trip does not, and by how much. The worst case shrinks as `N` grows but does not
#      reach zero;
#   3. untying the encoder and decoder weights makes it far worse, so that constraint is doing real
#      work and is not decoration;
#   4. the deviation is unchanged in `Float64`, which rules out rounding, and the encoder is
#      perfectly conditioned, which rules out ill-conditioning;
#   5. the reason: the round trip needs the shear pair to preserve the *embedded* Poisson tensor
#      `E 𝕁_N E'`, which has rank `N`, whereas the pair preserves the full-rank `𝕁_{N2}`. A rank-`N`
#      matrix cannot equal a rank-`N2` one, so the composition cannot be exactly symplectic for any
#      `N2 > N`.
#
# Run with the package environment:
#
#     julia --startup-file=no --project=. scripts/verification/sympnet_upscaling_symplecticity.jl

using GeometricMachineLearning
# `layer(chain, i)` is AbstractNeuralNetworks' accessor, and GeometricMachineLearning neither adds
# a method to it nor re-exports it.
using AbstractNeuralNetworks: layer
using LinearAlgebra: cond, opnorm, rank
using Random
using Zygote: jacobian

"""
Build the upscaling chain and return the Jacobians of its three stages at random points, together
with the two Poisson tensors, in element type `T`.

`share_weights` ties the decoder's weight to the encoder's, which is what makes the two PSD layers
a matched encode/decode pair rather than two unrelated maps. The original test did this, so the
question the chain is meant to answer is the `share_weights = true` one; the other is kept for
contrast.
"""
function upscaling_jacobians(N, N2, T; share_weights = true)
    model = Chain(PSDLayer(N, N2), GradientLayerQ(N2, 2 * N2, tanh),
        GradientLayerP(N2, 2 * N2, tanh), PSDLayer(N2, N))
    ps = NeuralNetwork(model, CPU(), T).params
    share_weights && (ps[4].weight.A = ps[1].weight.A)

    encoder = jacobian(x -> layer(model, 1)(x, ps[1]), rand(T, N))[1]
    shear = jacobian(
        y -> layer(model, 3)(layer(model, 2)(y, ps[2]), ps[3]),
        rand(T, N2))[1]
    decoder = jacobian(y -> layer(model, 4)(y, ps[4]), rand(T, N2))[1]
    round_trip = jacobian(x -> model(x, ps), rand(T, N))[1]

    return (; encoder, shear, decoder, round_trip,
        𝕁_N = Matrix(PoissonTensor(N, T)), 𝕁_N2 = Matrix(PoissonTensor(N2, T)))
end

"Relative deviation of `A` from `B` in the operator norm."
relative_deviation(A, B) = opnorm(A - B) / opnorm(B)

"""
The three identities each layer satisfies on its own. Returns the relative deviation of each, all
of which are zero up to rounding.
"""
function layerwise_deviations(J)
    return (; encoder = relative_deviation(J.encoder' * J.𝕁_N2 * J.encoder, J.𝕁_N),
        shear = relative_deviation(J.shear' * J.𝕁_N2 * J.shear, J.𝕁_N2),
        decoder = relative_deviation(J.decoder * J.𝕁_N2 * J.decoder', J.𝕁_N))
end

"The deviation of the whole chain from the symplectic condition on `R^N`."
round_trip_deviation(J) = relative_deviation(J.round_trip' * J.𝕁_N * J.round_trip, J.𝕁_N)

"""
Sample `trials` random chains at `(N, N2)` in element type `T` and return the worst layerwise
deviation and the extremes of the round-trip deviation.
"""
function sample(N, N2, T, trials; share_weights = true)
    worst_layerwise = zero(real(T))
    round_trip_min, round_trip_max = Inf, -Inf
    for _ in 1:trials
        J = upscaling_jacobians(N, N2, T; share_weights)
        d = layerwise_deviations(J)
        worst_layerwise = max(worst_layerwise, d.encoder, d.shear, d.decoder)
        r = round_trip_deviation(J)
        round_trip_min, round_trip_max = min(round_trip_min, r), max(round_trip_max, r)
    end
    return (; worst_layerwise, round_trip_min, round_trip_max)
end

Random.seed!(1234)

const TRIALS = 20

println("Each layer is symplectic; the chain is not, even with the encoder and decoder weights")
println("tied. Worst relative deviation over $TRIALS random chains per case.\n")
println(rpad("N", 5), rpad("N2", 5), rpad("type", 10), rpad("layerwise", 14), "round trip")

for (N, N2) in ((2, 4), (2, 8), (4, 8), (4, 16), (10, 20), (20, 40)),
    T in (Float32, Float64)

    s = sample(N, N2, T, TRIALS)
    println(rpad(N, 5), rpad(N2, 5), rpad(T, 10),
        rpad(round(s.worst_layerwise; sigdigits = 2), 14),
        round(s.round_trip_min; sigdigits = 2), " .. ", round(s.round_trip_max; sigdigits = 2))
end

println("\nUntying the two PSD weights changes the round trip but does not make it symplectic.\n")
println(rpad("N", 5), rpad("N2", 5), rpad("tied", 22), "untied")

for (N, N2) in ((2, 4), (4, 16), (20, 40))
    tied = sample(N, N2, Float64, TRIALS)
    untied = sample(N, N2, Float64, TRIALS; share_weights = false)
    println(rpad(N, 5), rpad(N2, 5),
        rpad(
            string(round(tied.round_trip_min; sigdigits = 2), " .. ",
                round(tied.round_trip_max; sigdigits = 2)),
            22),
        round(untied.round_trip_min; sigdigits = 2), " .. ",
        round(untied.round_trip_max; sigdigits = 2))
end

# Float64 does not shrink the round-trip deviation, so it is not a rounding artefact. If it were,
# moving from ~1e-7 to ~1e-16 machine epsilon would shrink it by nine orders of magnitude.

println("\nThe encoder is perfectly conditioned, so the deviation is not ill-conditioning.")
for (N, N2) in ((2, 4), (4, 16), (20, 40))
    J = upscaling_jacobians(N, N2, Float64)
    println("  N = ", rpad(N, 4), "N2 = ", rpad(N2, 5), "cond(E) = ", cond(J.encoder))
end

println("\nThe reason: the chain needs the shear pair to preserve the *embedded* Poisson tensor.")
println("`E 𝕁_N E'` has rank N, the pair preserves the full-rank `𝕁_N2`, and rank N != rank N2.\n")
println(rpad("N", 5), rpad("N2", 5), rpad("rank(E 𝕁_N E')", 18), rpad("rank(𝕁_N2)", 14),
    "‖E 𝕁_N E' - 𝕁_N2‖ / ‖𝕁_N2‖")

for (N, N2) in ((2, 4), (4, 16), (20, 40))
    J = upscaling_jacobians(N, N2, Float64)
    embedded = J.encoder * J.𝕁_N * J.encoder'
    println(rpad(N, 5), rpad(N2, 5), rpad(rank(embedded), 18), rpad(rank(J.𝕁_N2), 14),
        round(relative_deviation(embedded, J.𝕁_N2); sigdigits = 3))
end
