using SafeTestsets

@safetestset "Generalized Hamiltonian Neural Network" begin
    include("generalized_hamiltonian_neural_network_tests.jl")
end
@safetestset "Symbolic pullback for a single-layer PGHNN" begin
    include("pghnn_symbolic_pullback_single_layer_test.jl")
end
@safetestset "PGHNN training on a ParametricDataLoader" begin
    include("pghnn_training_test.jl")
end
@safetestset "Parametric and forced layers and architectures" begin
    include("parametric_layers_and_architectures_test.jl")
end
