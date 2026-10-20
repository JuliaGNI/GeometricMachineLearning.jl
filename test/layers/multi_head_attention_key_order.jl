using GeometricMachineLearning, Test
using GeometricMachineLearning: params
import Random

include("../helpers/eltypes.jl")

# The forward pass of `MultiHeadAttention` reads each head by its name, `head_1`, `head_2`, …, and
# stacks the heads in that order, whatever the order of the fields of the parameters.
@testset "MultiHeadAttention matches its heads by name ($T, $(tensor ? 3 : 2) axes)" for T in REAL_ELTYPES,
    tensor in (false, true)

    d = MultiHeadAttention(4, 2)
    x = tensor ? rand(Random.Xoshiro(7), T, 4, 3, 2) : rand(Random.Xoshiro(7), T, 4, 3)
    ps = params(NeuralNetwork(Chain(d), CPU(), T; rng = Random.Xoshiro(8))).L1
    swap(nt) = (head_2 = nt.head_2, head_1 = nt.head_1)
    y = d(x, ps)
    @test eltype(y) === T

    only_keys = (PQ = ps.PQ, PK = swap(ps.PK), PV = ps.PV)
    @test d(x, only_keys) == y

    every_projection = (PQ = swap(ps.PQ), PK = swap(ps.PK), PV = swap(ps.PV))
    @test d(x, every_projection) == y

    @test_throws "head_2" d(x, (PQ = ps.PQ, PK = (head_1 = ps.PK.head_1,), PV = ps.PV))
end
