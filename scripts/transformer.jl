using GeometricMachineLearning, LinearAlgebra
import Lux, Zygote, Random 

model = MultiHeadAttention(64, 8)
ps, st = Lux.setup(Random.default_rng(), model)
@time Lux.apply(model, rand(64, 4), ps, st);

@time Zygote.gradient(ps -> norm(Lux.apply(model, rand(64,4), ps, st)[1]), ps)[1];