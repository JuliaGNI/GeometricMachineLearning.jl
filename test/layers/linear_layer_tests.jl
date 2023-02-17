using GeometricMachineLearning
using Lux
using Random
using Test


Random.seed!(1234)

dummy_model = Linear(4,change_q=false)
ps,st = Lux.setup(Random.default_rng(), dummy_model)

@test dummy_model(ones(4),ps,st)[1] ≈ [1.0, 1.0, 1.2091209590435028, 0.34127573668956757]  atol = 1e-14
