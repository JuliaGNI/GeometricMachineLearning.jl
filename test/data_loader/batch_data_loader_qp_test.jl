using GeometricMachineLearning
using Test
import Random

Random.seed!(1234)

function dummy_qp_data_matrix(dim = 2, number_data_points = 200, T = Float32)
    @assert iseven(dim)
    (q = rand(T, dim ÷ 2, number_data_points), p = (rand(T, dim ÷ 2, number_data_points)))
end

function dummy_qp_data_tensor(dim = 2, number_of_time_steps = 100, number_of_parameters = 20, T = Float32)
    @assert iseven(dim)
    (q = rand(T, dim ÷ 2, number_of_time_steps, number_of_parameters),
        p = (rand(T, dim ÷ 2, number_of_time_steps, number_of_parameters)))
end

function test_data_loader(dim = 2, number_of_time_steps = 100,
        number_of_parameters = 20, batch_size = 10, n_epochs = 10, T = Float32)
    dl1 = DataLoader(dummy_qp_data_matrix(dim, number_of_time_steps, T))
    dl2 = DataLoader(dummy_qp_data_tensor(dim, number_of_time_steps, number_of_parameters, T))

    # A `(q, p)` pair is symplectic data, so the second axis is time and a pair of matrices is one
    # trajectory. A bare matrix reads its second axis as the parameter index instead, which is what
    # makes this path worth asserting: the same array shape means different things under the two
    # constructors.
    @test size(dl1.input.q) == (dim ÷ 2, number_of_time_steps, 1)
    @test (dl1.input_dim, dl1.input_time_steps, dl1.n_params) ==
          (dim, number_of_time_steps, 1)

    @test size(dl2.input.q) ==
          (dim ÷ 2, number_of_time_steps, number_of_parameters)
    @test (dl2.input_dim, dl2.input_time_steps, dl2.n_params) ==
          (dim, number_of_time_steps, number_of_parameters)

    arch1 = GSympNet(dl1)
    arch2 = GSympNet(dl2)

    nn1 = NeuralNetwork(arch1, CPU(), T)
    nn2 = NeuralNetwork(arch2, CPU(), T)

    batch = Batch(batch_size)
    o₁ = Optimizer(GradientOptimizer(), nn1)
    o₂ = Optimizer(GradientOptimizer(), nn2)

    params₁ = deepcopy(nn1.params)
    params₂ = deepcopy(nn2.params)

    loss_array₁ = o₁(nn1, dl1, batch, n_epochs; show_progress = false)
    loss_array₂ = o₂(nn2, dl2, batch, n_epochs; show_progress = false)

    # Both shapes reach the training loop and a step is taken for each. The assertion is that the
    # parameters moved, not that the loss fell: the data are random, so a fall over `n_epochs`
    # plain gradient steps is not reliable and an assertion on it would fail for a reason that has
    # nothing to do with the data loader.
    trained = ((loss_array₁, nn1, params₁), (loss_array₂, nn2, params₂))
    for (loss_array, nn, params) in trained
        @test length(loss_array) == n_epochs
        @test all(isfinite, loss_array)
        @test nn.params != params
    end
end

test_data_loader()
