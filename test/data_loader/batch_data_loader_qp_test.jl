using GeometricMachineLearning
using Test 

function dummy_qp_data_matrix(dim=2, number_data_points=200, T=Float32)
    (q = rand(T, dim, number_data_points), p = (rand(T, dim, number_data_points)))
end

function dummy_qp_data_tensor(dim=2, number_of_time_steps=100, number_of_parameters=20, T=Float32)
    (q = rand(T, dim, number_of_time_steps, number_of_parameters), p = (rand(T, dim, number_of_time_steps, number_of_parameters)))
end

function test_data_loader(dim=2, number_of_time_steps=100, number_of_parameters=20, batch_size, T=Float32)
    dl1 = DataLoader(dummy_qp_data_matrix(dim, number_of_time_steps, T))
    dl2 = DataLoader(dummy_qp_data_tensor(dim, number_of_time_steps, number_of_parameters))

    arch1 = GSympNet(dl1)
    arch2 = GSympNet(dl2)

    nn1 = NeuralNetwork(arch1, CPU(), T)
    nn2 = NeuralNetwork(arch2, CPU(), T)

    loss1 = loss(nn1, dl1)
    loss2 = loss(nn2, dl2)

    batch = Batch(batch_size)
    o₁ = Optimizer(GradientOptimizer(), nn1)
    o₂ = Optimizer(GradientOptimizer(), nn2)

    o₁(nn1, dl1, batch)
    o₂(nn2, dl2, batch)
end

test_data_loader()