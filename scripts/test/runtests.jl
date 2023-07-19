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

#=
I have implemented several new features in the code using new structures. These concern :
- the data,
- the training method (mainly reorganisation),
- new structures for preparing the training
- new structures for collecting training results,
- functions for creating HNN and LNN problems and for integrating a SympNet.

First of all, in the 'data' folder, there is now :
- batch.jl: it makes an appropriate batch for data with default values,
- data_symbol.jl : it contains tree types to correspond to the key of the data and the functionalities which depend on it,
- data_shape.jl: depending on whether the data are trajectories or simply sampled,
- data_training.jl: the main structure with the shape, data, symbol, problem, etc.

Then, to prepare the training sessions, there is now :
- training_paramters.jl: this contains the method, opt, batch size and nruns,
- training_set.jl: this contains a neural network, training parameters and data,
- ensemble_training.jl: brings together several training sets.
The train! function can be applied to all these structures.

Finally, training a neural network produces a new structure called NeuralNetSolution located in neural_net_solution.jl
which includes the nn, the problem, the neural network trained, the loss during the training process and a history of the structure which stores
all previous training of the neural network, i.e. you can apply train! directly to a NeuralNetSolution. 
EnsembleNeuralNetSolution brings together several NeuralNetSolutions, resulting for example from the application of train! to an ensemble training.
ensemble training.

I have created my own test folder with its own runtests.jl which should be merged with the existing one (but I don't know if it's up to date or not).
Several minor tests are missing such as:
- check_batch_size functionality,
- architecture,
and probably other features I haven't seen. My own test folder is located in "script".
=#