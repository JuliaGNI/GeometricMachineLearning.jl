using GeometricMachineLearning, Test
import Random

Random.seed!(1234)

function test_set_up_and_application(T = Float32, sys_dim = 49, output_dim = 10,
        seq_length = 16, batch_size = 32; average = false)
    d = Chain(ClassificationLayer(sys_dim, output_dim, σ, average = average))
    ps = NeuralNetwork(d, CPU(), T).params
    output₁ = d(rand(T, sys_dim, seq_length), ps)
    output₂ = d(rand(T, sys_dim, seq_length, batch_size), ps)
    @test size(output₁) == (10, 1)
    @test size(output₂) == (10, 1, batch_size)
end

test_set_up_and_application(average = false)
test_set_up_and_application(average = true)

l = ClassificationLayer(2, 2, identity; average = true)
ps = (weight = [1 0; 0 1],)
input = [1 2 3; 1 1 1]
@test l(input, ps) == reshape([2.0, 1.0], 2, 1)
l = ClassificationLayer(2, 2, identity; average = false)
@test l(input, ps) == reshape([3, 1], 2, 1)
