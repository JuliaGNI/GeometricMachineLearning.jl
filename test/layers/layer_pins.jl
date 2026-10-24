# The forward pass of each layer whose methods share code across a `Q`/`P` pair, a `Bool` choice or
# a triangle, against the output that commit 4c4a8320 records for the same seed and input. The
# cases cover dense, `SymmetricMatrix`, triangular, `SkewSymMatrix` and Stiefel leaves; the helper
# file says how the pins are kept independent of the platform.

using GeometricMachineLearning
using Test

include("../helpers/eltypes.jl")
include("../helpers/pins.jl")
include("../helpers/pinned_outputs.jl")

@testset "$name ($T)" for (i, (name, model, dim, cols)) in enumerate(layer_pin_cases()),
    T in REAL_ELTYPES

    nn = pinned_network(model, T, layer_seed(i))
    y = nn(pin_input(T, dim, cols))
    @test eltype(y) === T
    @test pin_ulps(vec(y), LAYER_PINS[(name, T)]) ≤ PIN_ULPS
end

# The comparison itself: its scale is the largest magnitude of the pin, not the entry's own, and a
# deviation of 16 of those ULPs passes where 17 fail.
@testset "pin_ulps counts in ULPs of the largest magnitude ($T)" for T in REAL_ELTYPES
    x = T[1.5, -0.25, 3]
    shifted(n) = [x[1], x[2] + n * eps(T(3)), x[3]]
    @test pin_ulps(shifted(16), x) == 16
    @test pin_ulps(shifted(16), x) ≤ PIN_ULPS
    @test !(pin_ulps(shifted(17), x) ≤ PIN_ULPS)
    @test pin_ulps(x[1:2], x) == Inf
end
