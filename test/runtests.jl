
using SafeTestsets

@safetestset "Arrays                                                                          " begin include("array_tests.jl") end
@safetestset "Abstract Layer                                                                  " begin include("abstract_layer_tests.jl") end
@safetestset "Feed Forward Layer                                                              " begin include("feed_forward_layer_tests.jl") end
@safetestset "Residual Layer                                                                  " begin include("residual_layer_tests.jl") end
@safetestset "Symplectic Layers                                                               " begin include("symplectic_layer_tests.jl") end
@safetestset "Abstract Neural Network                                                         " begin include("abstract_neural_network_tests.jl") end
@safetestset "Vanilla Neural Network                                                          " begin include("vanilla_neural_network_tests.jl") end
@safetestset "Hamiltonian Neural Network                                                      " begin include("hamiltonian_neural_network_tests.jl") end
