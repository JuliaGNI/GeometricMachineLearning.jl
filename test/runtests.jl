
using SafeTestsets

@safetestset "Arrays                                                                          " begin include("arrays/array_tests.jl") end
#@safetestset "Gradient Layer                                                                  " begin include("layers/gradient_layer_tests.jl") end
@safetestset "Symplectic Layers                                                               " begin include("layers/symplectic_layer_tests.jl") end
@safetestset "Hamiltonian Neural Network                                                      " begin include("hamiltonian_neural_network_tests.jl") end
#@safetestset "Custom AD rules for kernels                                                     " begin include("custom_ad_rules/kernel_pullbacks.jl") end
#@safetestset "Transformer Networks #1                                                         " begin include("transformer_related/multi_head_attention_stiefel_optim_cache.jl") end
#@safetestset "Transformer Networks #2                                                         " begin include("transformer_related/multi_head_attention_stiefel_retraction.jl") end
#@safetestset "Transformer Networks #3                                                         " begin include("transformer_related/multi_head_attention_stiefel_setup.jl") end
@safetestset "Optimizer #1                                                                    " begin include("optimizers/utils/global_sections.jl") end
#@safetestset "Optimizer #2                                                                    " begin include("optimizers/utils/optimization_step.jl") end
@safetestset "Optimizer #3                                                                    " begin include("optimizers/utils/modified_exponential.jl") end
