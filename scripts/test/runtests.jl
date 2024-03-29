using GeometricMachineLearning
using Test
using SafeTestsets

include("data_generation.jl")
include("macro_testerror.jl")

@safetestset "Architecture                                                                    " begin include("test_architecture.jl") end
@safetestset "Data                                                                            " begin include("test_data.jl") end
@safetestset "Batch                                                                           " begin include("test_batch.jl") end
@safetestset "Method                                                                          " begin include("test_method.jl") end
@safetestset "Matching                                                                        " begin include("test_matching.jl") end
@safetestset "TrainingSet                                                                     " begin include("test_trainingSet.jl") end
@safetestset "Training                                                                        " begin include("test_training.jl") end
@safetestset "NeuralNetSolution                                                               " begin include("test_neuralnet_solution.jl") end
@safetestset "Problem & Integrators                                                           " begin include("test_integrator.jl") end
