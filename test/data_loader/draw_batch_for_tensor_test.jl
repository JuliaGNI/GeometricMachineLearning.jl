using GeometricMachineLearning, Test 
using GeometricMachineLearning: convert_input_and_batch_indices_to_array
import Random 

Random.seed!(1234)

function test_data_loader_for_qp_tensor(system_dim2::Int, input_time_steps::Int, n_params::Int, batch_size::Int, seq_length::Int, prediction_window::Union{Int, Nothing}=nothing)
    _prediction_window = isnothing(prediction_window) ? seq_length : prediction_window
    dummy_data = (q = rand(system_dim2, input_time_steps, n_params), p = rand(system_dim2, input_time_steps, n_params))

    dl = DataLoader(dummy_data)
    batch = isnothing(prediction_window) ? Batch(batch_size, seq_length) : Batch(batch_size, seq_length, prediction_window)

    batch_indices_all = batch(dl)
    for batch_indices in batch_indices_all
        batched_array = convert_input_and_batch_indices_to_array(dl, batch, batch_indices)
        @test size(batched_array[1].q) == (system_dim2, seq_length,         batch_size)
        @test size(batched_array[2].q) == (system_dim2, _prediction_window, batch_size)
    end
end

test_data_loader_for_qp_tensor(5, 50, 20, 10, 16)