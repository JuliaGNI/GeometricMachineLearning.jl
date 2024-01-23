using GeometricMachineLearning, Test 
using GeometricMachineLearning: convert_input_and_batch_indices_to_array

function test_data_loader_for_qp_tensor(system_dim2::Int, input_time_steps::Int, n_params::Int, batch_size::Int, seq_length::Int)
    dummy_data = (q = rand(system_dim2, input_time_steps, n_params), p = rand(system_dim2, input_time_steps, n_params))

    dl = DataLoader(dummy_data)
    batch = Batch(batch_size, seq_length)

    batch_indices_all = batch(dl)
    for batch_indices in batch_indices_all
        batched_array = convert_input_and_batch_indices_to_array(dl, batch, batch_indices)
        @test size(batched_array[1].q) == (system_dim2, seq_length, batch_size)
        @test size(batched_array[2].q) == (system_dim2, 1,          batch_size)
    end
end

test_data_loader_for_qp_tensor(5, 50, 20, 10, 16)