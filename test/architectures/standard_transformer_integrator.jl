using GeometricMachineLearning
using Test
import Random

Random.seed!(123)

const sys_dim = 5
const model = StandardTransformerIntegrator(sys_dim)
const seq_length = 5

function set_up_and_apply_integrator(; T = Float32)
    ics = rand(T, sys_dim, seq_length)

    nn = NeuralNetwork(model, CPU(), T)

    iterates = integrate(nn, ics)

    @test ics ≉ iterates[:, (end - seq_length + 1):end]
end

set_up_and_apply_integrator(; T = Float32)
set_up_and_apply_integrator(; T = Float64)

# The two transformer families disagree on where the sequence length comes from. A
# `StandardTransformerIntegrator` takes the width of the input it is handed; a
# `VolumePreservingTransformer` stores one, and cannot take an input of another width -- so the
# branch is asserted directly rather than through the integrator.
@test GeometricMachineLearning._seq_length(StandardTransformerIntegrator(sys_dim),
    rand(Float32, sys_dim, 5)) == 5
@test GeometricMachineLearning._seq_length(SymplecticTransformer(sys_dim),
    rand(Float32, sys_dim, 5)) == 5
@test GeometricMachineLearning._seq_length(VolumePreservingTransformer(4, 3),
    rand(Float32, 4, 5)) == 3
