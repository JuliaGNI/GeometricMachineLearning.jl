using GeometricMachineLearning, Metal, Test

@test check(rand(MetalBackend(), StiefelManifold{Float32}, 50, 10)) < .1f0