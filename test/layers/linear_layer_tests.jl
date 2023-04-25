using GeometricMachineLearning
using Lux
using Random
using Test


Random.seed!(1234)

dummy_model = Linear(4,change_q=false)
ps,st = Lux.setup(Random.default_rng(), dummy_model)

@test dummy_model(ones(4),ps,st)[1] ≈ [1.0, 1.0, 1.1045604944229126, 0.670637883245945]  atol = 1e-14
