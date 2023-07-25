using GeometricMachineLearning
using Lux
using Random
using Test

Random.seed!(1234)

dummy_model = Gradient(4,6,tanh,change_q=false)
ps,st = Lux.setup(Random.default_rng(), dummy_model)

@test dummy_model(ones(4),parameters)[1] ≈ [1.0; 1.0; 1.0790170586786156; 1.2189119955519803;;]  atol = 1e-14
