using GeometricMachineLearning, Pkg

Pkg.add("Metal")
Pkg.add("Test")
using Metal, Test

@test check(rand(StiefelManifold{Float32}, MetalBackend(), 50, 10)) < .1f0