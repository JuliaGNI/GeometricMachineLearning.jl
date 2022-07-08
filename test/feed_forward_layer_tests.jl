using GeometricMachineLearning
using Test


l = FeedForwardLayer(x -> x, ones(2,2), ones(2), x -> 1)
i = ones(2)
o1 = zero(i)
o2 = zero(i)

@test l(o1, i) == apply!(o2, i, l) == 3*i
