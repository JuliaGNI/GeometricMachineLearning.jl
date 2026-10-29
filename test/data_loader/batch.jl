# The `Batch` method of `integrate` exists to carry one number: it is the three-argument form a
# training script writes, and it has to give the trajectory the keyword form gives. The window is
# chosen so that the two disagree when it is dropped: a standard transformer of system dimension
# 5 on a four-column input takes two predictions of two columns for `n_points = 8`, where the
# default window takes one of four columns. Both forms carry the input's element type, so a
# trajectory copied to the host or plotted keeps its precision.

using GeometricMachineLearning
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

@testset "the Batch method of integrate forwards its prediction window ($T)" for T in REAL_ELTYPES
    nn = NeuralNetwork(StandardTransformerIntegrator(5), CPU(), T)
    ics = rand(T, 5, 4)

    @test eltype(integrate(nn, ics, Batch(3, 5, 2); n_points = 5)) === T
    @test eltype(integrate(nn, ics; n_points = 5, prediction_window = 2)) === T
    @test integrate(nn, ics, Batch(3, 5, 2); n_points = 5) ==
          integrate(nn, ics; n_points = 5, prediction_window = 2)
    @test integrate(nn, ics, Batch(3, 5, 2); n_points = 8) ==
          integrate(nn, ics; n_points = 8, prediction_window = 2)
    @test integrate(nn, ics, Batch(3, 5, 2); n_points = 8) ≠
          integrate(nn, ics; n_points = 8)
end

# The three-argument method is a method of `GeometricBase.integrate` on `ReducedComplexityModeling`'s
# `Batch`, so its `nn` names this package's `TransformerIntegrator`; a signature without that
# annotation is piracy. `Aqua`'s check does not see this one: it treats a symbol type parameter as a
# type of the package's own, so no signature written on `Batch{:Transformer}` reads as foreign. That
# is why the narrowing has a gate here and the piracies Aqua does see have one in
# `test/quality/aqua.jl`.
@testset "the Batch method of integrate is narrowed to a TransformerIntegrator" begin
    @test only(methods(integrate, (NeuralNetwork, Any, Batch{:Transformer}))).sig <:
          Tuple{typeof(integrate), NeuralNetwork{<:TransformerIntegrator}, Any, Any}
end
