using GeometricMachineLearning
using Test
import Random 

Random.seed!(123)

const sys_dim = 5
const model = RegularTransformerIntegrator(sys_dim)
const seq_length = 5

function set_up_and_apply_integrator(; T=Float32)
    ics = rand(T, sys_dim, seq_length)

    nn = NeuralNetwork(model, CPU(), T)

    iterates = iterate(nn, ics)

    @test ics ≉ iterates[:, (end - seq_length + 1):end]
end

set_up_and_apply_integrator(; T = Float32)
set_up_and_apply_integrator(; T = Float64)