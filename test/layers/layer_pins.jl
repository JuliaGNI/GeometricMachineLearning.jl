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
    @test within_ulps(vec(y), LAYER_PINS[(name, T)])
end

# The comparison itself: it holds at 8 ULPs and fails beyond them.
@testset "within_ulps separates 8 ULPs from 9 ($T)" for T in REAL_ELTYPES
    x = T[1.5, -0.25, 3]
    @test within_ulps(x, x .+ 8 .* eps.(x))
    @test !within_ulps(x, x .+ 9 .* eps.(x))
    @test !within_ulps(x, x[1:2])
end
