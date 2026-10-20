# The type arguments of the layers: a `GradientLayer` stores its dimensions as `Int`, whatever
# integer type it is given, in a concrete field; and every method under `src/layers/` that takes an
# element type takes it as `::Type{T}`, so that the method specialises on it.

using GeometricMachineLearning
using GeometricMachineLearning: GradientLayer
using Test

@testset "GradientLayer stores its dimensions as Int ($(typeof(M)), $(typeof(n)))" for M in (
        4, Int32(4)),
    n in (8, Int32(8))

    for constructor in (GradientLayerQ, GradientLayerP)
        l = constructor(M, n, tanh)
        @test isconcretetype(fieldtype(typeof(l), :second_dim))
        @test l.second_dim === 8
        @test typeof(l) === typeof(constructor(4, 8, tanh))
    end
end

@testset "no method under src/layers/ takes an element type as T::Type" begin
    directory = joinpath(pkgdir(GeometricMachineLearning), "src", "layers")
    hits = [(file, i)
            for file in readdir(directory; join = true)
            for (i, line) in enumerate(eachline(file)) if occursin(r"T::Type\b", line)]
    @test isempty(hits)
end
