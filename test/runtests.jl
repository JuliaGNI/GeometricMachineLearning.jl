using SafeTestsets, Test, GeometricMachineLearning

# A test that trains passes `show_progress = false`. The `Optimizer` functor defaults it to `true`,
# which is right at a REPL and is noise in a suite -- a 2048-epoch run emits a few hundred progress
# lines and buries the failure you are looking for. `train!` already defaults `showprogress = false`,
# so only the functor needs saying.

# reduced order modeling tests
@info "Starting reduced-order-modeling tests"
@safetestset "PSD tests                                                                       " begin
    include("psd_architecture_tests.jl")
end
@safetestset "SymplecticAutoencoder tests                                                     " begin
    include("symplectic_autoencoder_tests.jl")
end
@safetestset "Check if autoencoder error is lower than PSD error                              " begin
    include("sae_error_lower_than_psd_error.jl")
end
@safetestset "Check reduced model                                                             " begin
    include("reduced_system.jl")
end
@safetestset "Check parameterlength                                                           " begin
    include("parameterlength/check_parameterlengths.jl")
end
@safetestset "Symplectic Potential (array tests)                                              " begin
    include("arrays/poisson_tensor.jl")
end
@safetestset "Test triangular matrices                                                        " begin
    include("arrays/triangular.jl")
end
@safetestset "Gradient Layer                                                                  " begin
    include("layers/gradient_layer_tests.jl")
end
@safetestset "Test symplecticity of upscaling layer                                           " begin
    include("layers/sympnet_layers_test.jl")
end
@safetestset "Hamiltonian Neural Network                                                      " begin
    include("hamiltonian_neural_network_tests.jl")
end
@safetestset "Generalized Hamiltonian Neural Network                                          " begin
    include("generalized_hamiltonian_neural_networks_test.jl")
end
@safetestset "Symbolic pullback for a single-layer PGHNN                                      " begin
    include("generalized_hamiltonian_neural_networks/pghnn_symbolic_pullback_single_layer_test.jl")
end
@safetestset "PGHNN training on a ParametricDataLoader                                        " begin
    include("generalized_hamiltonian_neural_networks/pghnn_training_test.jl")
end
@safetestset "Manifold Neural Network Layers                                                  " begin
    include("layers/manifold_layers.jl")
end
@safetestset "Custom tensor matrix multiplication                                             " begin
    include("kernels/tensor_mat_mul.jl")
end
@safetestset "Custom inverse for 2x2, 3x3, 4x4, 5x5 matrices                                  " begin
    include("kernels/tensor_inverse.jl")
end
@safetestset "Custom AD rules for kernels                                                     " begin
    include("custom_ad_rules/kernel_pullbacks.jl")
end
@safetestset "ResNet                                                                          " begin
    include("layers/resnet_tests.jl")
end
# transformer-related tests
@info "Starting transformer-related tests"
@safetestset "Test setup of MultiHeadAttention layer Stiefel weights                          " begin
    include("transformer_related/multi_head_attention_stiefel_setup.jl")
end
@safetestset "Test geodesic and Cayley retr for the MultiHeadAttention layer w/ St weights    " begin
    include("transformer_related/multi_head_attention_stiefel_retraction.jl")
end
@safetestset "Test the correct setup of the various optimizer caches for MultiHeadAttention   " begin
    include("transformer_related/multi_head_attention_stiefel_optim_cache.jl")
end
@safetestset "Check if the transformer can be applied to a tensor.                            " begin
    include("transformer_related/transformer_application.jl")
end
@safetestset "Check if the gradient/pullback of MultiHeadAttention changes type in St case    " begin
    include("transformer_related/transformer_gradient.jl")
end
@safetestset "Check if the optimization_step! changes the parameters of the transformer       " begin
    include("transformer_related/transformer_optimizer.jl")
end

@safetestset "Attention layer #1                                                              " begin
    include("attention_layer/attention_setup.jl")
end
@safetestset "Classification layer                                                            " begin
    include("layers/classification.jl")
end
@info "Starting optimizer tests"
@safetestset "Optimizer #2                                                                    " begin
    include("optimizers/utils/optimization_step.jl")
end
@safetestset "Optimizer #3                                                                    " begin
    include("optimizers/optimizer_convergence_tests/svd_optim.jl")
end
@safetestset "Optimizer #4                                                                    " begin
    include("optimizers/optimizer_convergence_tests/psd_optim.jl")
end
@safetestset "Check if Adam with decay converges                                              " begin
    include("optimizers/optimizer_convergence_tests/adam_with_learning_rate_decay.jl")
end
@safetestset "Gradient optimizer tests                                                        " begin
    include("optimizers/gradient_optimizer.jl")
end
@safetestset "Momentum optimizer tests                                                        " begin
    include("optimizers/momentum_optimizer.jl")
end
@safetestset "Optimizers with structured (non-manifold) weights                               " begin
    include("optimizers/structured_array_parameters.jl")
end
@info "Starting data and data-loader tests"
@safetestset "Data                                                                            " begin
    include("data/test_data.jl")
end
@safetestset "Batch                                                                           " begin
    include("data/test_batch.jl")
end
# @safetestset "Method                                                                          " begin include("train!/test_method.jl") end
@safetestset "Matching                                                                        " begin
    include("data/test_matching.jl")
end

@safetestset "Test data loader for q and p data                                               " begin
    include("data_loader/batch_data_loader_qp_test.jl")
end
@safetestset "TrainingParameters and the step size passed to train!                           " begin
    include("training_parameters.jl")
end
@safetestset "Test the data loader in combination with optimization_step!                     " begin
    include("data_loader/data_loader_optimization_step.jl")
end
@safetestset "Optimizer functor with data loader for Adam                                     " begin
    include("data_loader/optimizer_functor_with_adam.jl")
end
@safetestset "Test data loader for a tensor (q and p data)                                    " begin
    include("data_loader/draw_batch_for_tensor_test.jl")
end
@safetestset "Parametric DataLoader                                                           " begin
    include("data_loader/parametric_data_loader_test.jl")
end

@info "Starting network-loss and kernel tests"
@safetestset "Test NetworkLoss + Optimizer                                                    " begin
    include("network_losses/losses_and_optimization.jl")
end

@info "Starting integrator and attention tests"
@safetestset "Test parallel inverses                                                          " begin
    include("kernels/tensor_inverse.jl")
end
@safetestset "Test parallel Cayley                                                            " begin
    include("kernels/tensor_cayley.jl")
end

@safetestset "Test volume-preserving feedforward neural network                               " begin
    include("layers/volume_preserving_feedforward.jl")
end

@safetestset "SympNet integrator                                                              " begin
    include("sympnet_integrator.jl")
end
@safetestset "Regular transformer integrator                                                  " begin
    include("standard_transformer_integrator.jl")
end

@safetestset "Batch functor(s)                                                                " begin
    include("batch/batch_functor.jl")
end

@safetestset "Volume-Preserving Transformer (skew-symmetric tests)                            " begin
    include("volume_preserving_attention/test_skew_map.jl")
end
@safetestset "Volume-Preserving Transformer (cayley-transform tests)                          " begin
    include("volume_preserving_attention/test_cayley_transforms.jl")
end

@safetestset "Linear Symplectic Attention                                                     " begin
    include("linear_symplectic_attention.jl")
end
@safetestset "Linear Symplectic Transformer                                                   " begin
    include("linear_symplectic_transformer.jl")
end

@info "Starting final data-loader and documentation tests"
@safetestset "DataLoader for input and output                                                 " begin
    include("data_loader/data_loader_for_input_and_output.jl")
end

@safetestset "HDF5 save/load for GML special array types                                     " begin
    include("hdf5_support.jl")
end
@safetestset "Data loader docstring examples                                                  " begin
    include("docstrings/data_loader.jl")
end
@safetestset "Layer and architecture docstring examples                                      " begin
    include("docstrings/layers_and_architectures.jl")
end
@safetestset "Loss docstring examples                                                        " begin
    include("docstrings/losses.jl")
end
@safetestset "Manifold docstring examples                                                    " begin
    include("docstrings/manifolds.jl")
end
@safetestset "Utility and pullback docstring examples                                        " begin
    include("docstrings/utilities.jl")
end
