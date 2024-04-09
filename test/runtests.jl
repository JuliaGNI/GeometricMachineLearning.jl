
using SafeTestsets

@safetestset "Check parameterlength                                                           " begin include("parameterlength/check_parameterlengths.jl") end
@safetestset "Arrays #1                                                                       " begin include("arrays/array_tests.jl") end
@safetestset "Sampling of arrays                                                              " begin include("arrays/random_generation_of_custom_arrays.jl") end
@safetestset "Addition tests for custom arrays                                                " begin include("arrays/addition_tests_for_custom_arrays.jl") end
@safetestset "Scalar multiplication tests for custom arrays                                   " begin include("arrays/scalar_multiplication_for_custom_arrays.jl") end
@safetestset "Matrix multiplication tests for custom arrays                                   " begin include("arrays/matrix_multiplication_for_custom_arrays.jl") end
@safetestset "Test constructors for custom arrays                                             " begin include("arrays/constructor_tests_for_custom_arrays.jl") end
@safetestset "Manifolds (Grassmann):                                                          " begin include("manifolds/grassmann_manifold.jl") end
@safetestset "Gradient Layer                                                                  " begin include("layers/gradient_layer_tests.jl") end
@safetestset "Test symplecticity of upscaling layer                                           " begin include("layers/sympnet_layers_test.jl") end 
@safetestset "Hamiltonian Neural Network                                                      " begin include("hamiltonian_neural_network_tests.jl") end
@safetestset "Manifold Neural Network Layers                                                  " begin include("layers/manifold_layers.jl") end
@safetestset "Custom AD rules for kernels                                                     " begin include("custom_ad_rules/kernel_pullbacks.jl") end
@safetestset "ResNet                                                                          " begin include("layers/resnet_tests.jl") end

# transformer-related tests
@safetestset "Test setup of MultiHeadAttention layer Stiefel weights                          " begin include("transformer_related/multi_head_attention_stiefel_setup.jl") end
@safetestset "Test geodesic and Cayley retr for the MultiHeadAttention layer w/ St weights    " begin include("transformer_related/multi_head_attention_stiefel_retraction.jl") end
@safetestset "Test the correct setup of the various optimizer caches for MultiHeadAttention   " begin include("transformer_related/multi_head_attention_stiefel_optim_cache.jl") end
@safetestset "Check if the transformer can be applied to a tensor.                            " begin include("transformer_related/transformer_application.jl") end
@safetestset "Check if the gradient/pullback of MultiHeadAttention changes type in St case    " begin include("transformer_related/transformer_gradient.jl") end
@safetestset "Check if the optimization_step! changes the parameters of the transformer       " begin include("transformer_related/transformer_optimizer.jl") end

@safetestset "Attention layer #1                                                              " begin include("attention_layer/attention_setup.jl") end
@safetestset "Classification layer                                                            " begin include("layers/classification.jl") end
@safetestset "Optimizer #1                                                                    " begin include("optimizers/utils/global_sections.jl") end
@safetestset "Optimizer #2                                                                    " begin include("optimizers/utils/optimization_step.jl") end
@safetestset "Optimizer #3                                                                    " begin include("optimizers/utils/modified_exponential.jl") end
@safetestset "Optimizer #4                                                                    " begin include("optimizers/optimizer_convergence_tests/svd_optim.jl") end
@safetestset "Optimizer #5                                                                    " begin include("optimizers/optimizer_convergence_tests/psd_optim.jl") end
@safetestset "Data                                                                            " begin include("data/test_data.jl") end
@safetestset "Batch                                                                           " begin include("data/test_batch.jl") end
@safetestset "Method                                                                          " begin include("train!/test_method.jl") end
@safetestset "Matching                                                                        " begin include("data/test_matching.jl") end
@safetestset "TrainingSet                                                                     " begin include("train!/test_trainingSet.jl") end
@safetestset "Training                                                                        " begin include("train!/test_training.jl") end
@safetestset "NeuralNetSolution                                                               " begin include("train!/test_neuralnet_solution.jl") end
@safetestset "Problem & Integrators                                                           " begin include("integrator/test_integrator.jl") end

@safetestset "Test data loader for q and p data                                               " begin include("data_loader/batch_data_loader_qp_test.jl") end
@safetestset "Test mnist_utils.                                                               " begin include("data_loader/mnist_utils.jl") end
@safetestset "Test the data loader in combination with optimization_step!                     " begin include("data_loader/data_loader_optimization_step.jl") end
@safetestset "Optimizer functor with data loader for Adam                                     " begin include("data_loader/optimizer_functor_with_adam.jl") end
@safetestset "Test data loader for a tensor (q and p data)                                    " begin include("data_loader/draw_batch_for_tensor_test.jl") end

@safetestset "Test parallel inverses                                                          " begin include("kernels/tensor_inverse.jl") end
@safetestset "Test parallel Cayley                                                            " begin include("kernels/tensor_cayley.jl") end