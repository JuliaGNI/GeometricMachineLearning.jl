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
I implemented severals new features in the code through new structures. It deals with :
- data,
- training method,
- new structures to prepare training,
- new structures to gathers the results of training,
- some functions to create HNN problem, LNN problem and to integrate a SympNet.

First, in the folder "data", there is now :
- batch.jl : it does appropriately batch for data with default values,
- data_symbol.jl : it contains tree types to match key of data and functionalities which depend on,
- data_shape.jl : depending on if data are trajectories or just sampled,
- data_training.jl : the big structure with shape, data, symbol, problem, etc.

Then, to prepare trainings, there is now :
- training_paramters.jl : it gathers method, opt, batch size, nruns,
- training_set.jl : it gathers a neural network, a training parameters, and data,
- ensemble_training.jl : it gathers severals training sets.
The train! function can be applied on all of those structures.

Finally, when we train a neural network, it gives a new structure called NeuralNetSolution localized in neural_net_solution.jl
which includes the nn, the problem, the trained neural network, the loss during the training process and a structure history wich memories
all the previous training on the neural network, that is to we can apply train! directly on a NeuralNetSolution. 
EnsembleNeuralNetSolution gathers severals NeuralNetSolution, as a result for example of the application of train! on an
ensemble training.

I created my own test folder with its own runtests.jl that should be merged in the existing one (but I don't know if it is up to date or not).
Severals minors tests are missing like :
- the check_batch_size functionality,
- Architecture,
and probaby others functionality that I didn't see. My own test folder is located in "script".
=#