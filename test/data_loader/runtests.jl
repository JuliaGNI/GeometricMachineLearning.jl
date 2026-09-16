using SafeTestsets

@safetestset "Test data loader for q and p data" begin
    include("batch_data_loader_qp_test.jl")
end
@safetestset "Test the data loader in combination with optimization_step!" begin
    include("data_loader_optimization_step.jl")
end
@safetestset "Optimizer functor with data loader for Adam" begin
    include("optimizer_functor_with_adam.jl")
end
@safetestset "Test data loader for a tensor (q and p data)" begin
    include("draw_batch_for_tensor_test.jl")
end
@safetestset "Batch functor(s)" begin
    include("batch_functor.jl")
end
@safetestset "DataLoader for input and output" begin
    include("data_loader_for_input_and_output.jl")
end
